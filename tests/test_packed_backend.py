from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
except ImportError:
    np = pa = pq = torch = Image = None


@unittest.skipIf(torch is None, "packed backend dependencies are not installed")
class PackedBackendTests(unittest.TestCase):
    def test_packed_decode_matches_png_pixels(self) -> None:
        from game_cls.data.image_spec import ImageSpec
        from game_cls.data.lazy_pair_dataset import (
            LazyTrainingPairDataset,
            PairRequest,
            build_eval_dataset,
        )
        from game_cls.data.packed_backend import (
            PackedUint8Backend,
            pack_frame_index,
        )
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            expected = []
            for index in range(5):
                array = np.full((208, 448, 3), index * 40, dtype=np.uint8)
                array[:, :, 1] += 5
                path = root / f"image_{index}.png"
                Image.fromarray(array).save(path)
                rows.append(
                    {
                        "path": str(path),
                        "game": "game_A",
                        "label": index % 2,
                        "video_id": f"{index % 2:02d}",
                        "frame_id": index,
                    }
                )
                expected.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
            frame_index = root / "frames.parquet"
            pq.write_table(pa.Table.from_pylist(rows), frame_index)
            image_spec = ImageSpec(width=448, height=208, channels=3)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                image_spec=image_spec,
                images_per_shard=2,
            )
            backend = PackedUint8Backend(
                packed_index,
                image_spec=image_spec,
                max_open_shards=1,
            )
            for location, tensor in enumerate(expected):
                self.assertTrue(torch.equal(backend(location), tensor))
                self.assertLessEqual(backend.open_shard_count, 1)
            reordered = backend.get_many([4, 0, 3, 1])
            self.assertEqual(tuple(reordered.shape), (4, 3, 208, 448))
            for actual, location in zip(reordered, [4, 0, 3, 1], strict=False):
                self.assertTrue(torch.equal(actual, expected[location]))
            self.assertEqual(len(list((root / "packed").glob("shard_*.bin"))), 3)
            self.assertTrue(
                (root / "packed" / "packed_video_entries.parquet").is_file()
            )
            videos = read_video_entries_parquet(
                root / "packed" / "packed_video_entries.parquet",
                deltas=(2,),
            )
            dataset = build_eval_dataset(videos, 2, decoder=backend)
            sample = dataset[0]
            self.assertEqual(tuple(sample["images"].shape), (2, 3, 208, 448))
            batch_samples = dataset.__getitems__([0, 1])
            self.assertEqual(len(batch_samples), 2)
            self.assertEqual(
                tuple(batch_samples[0]["images"].shape),
                (2, 3, 208, 448),
            )
            train_dataset = LazyTrainingPairDataset(videos, decoder=backend)
            train_samples = train_dataset.__getitems__(
                [
                    PairRequest(0, 2, 0, augmentation_seed=1),
                    PairRequest(0, 2, 1, augmentation_seed=2),
                ]
            )
            self.assertEqual(len(train_samples), 2)
            self.assertEqual(
                tuple(train_samples[0]["images"].shape),
                (2, 3, 208, 448),
            )
            self.assertTrue(sample["meta"]["image0_path"].startswith("packed://frame/"))
            manifest_path = root / "packed" / "packed_manifest.json"
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertNotIn(str((root / "packed").resolve()), manifest_text)
            import json

            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["width"], 448)
            self.assertEqual(manifest["height"], 208)
            self.assertEqual(manifest["channels"], 3)
            with self.assertRaisesRegex(ValueError, "does not match configured shape"):
                PackedUint8Backend(
                    packed_index,
                    image_spec=ImageSpec(width=208, height=448, channels=3),
                )
            backend.close()

    def test_packed_dataset_spawn_worker_owns_its_memmap(self) -> None:
        from torch.utils.data import DataLoader

        from game_cls.data.collate import pair_collate
        from game_cls.data.image_spec import ImageSpec
        from game_cls.data.lazy_pair_dataset import build_eval_dataset
        from game_cls.data.packed_backend import (
            PackedUint8Backend,
            pack_frame_index,
        )
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for frame_id in range(3):
                array = np.full((8, 8, 3), frame_id * 50, dtype=np.uint8)
                path = root / f"01{frame_id:05d}.png"
                Image.fromarray(array).save(path)
                rows.append(
                    {
                        "path": str(path),
                        "game": "game_A",
                        "label": 0,
                        "video_id": "01",
                        "frame_id": frame_id,
                    }
                )
            frame_index = root / "frames.parquet"
            pq.write_table(pa.Table.from_pylist(rows), frame_index)
            image_spec = ImageSpec(width=8, height=8, channels=3)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                image_spec=image_spec,
                images_per_shard=2,
            )
            backend = PackedUint8Backend(
                packed_index,
                image_spec=image_spec,
                max_open_shards=1,
            )
            videos = read_video_entries_parquet(
                root / "packed" / "packed_video_entries.parquet",
                deltas=(1,),
            )
            dataset = build_eval_dataset(videos, 1, decoder=backend)
            loader = DataLoader(
                dataset,
                batch_size=2,
                num_workers=1,
                multiprocessing_context="spawn",
                timeout=20,
                collate_fn=pair_collate,
            )

            batches = list(loader)

            self.assertEqual(len(batches), 1)
            self.assertEqual(tuple(batches[0]["images"].shape), (2, 2, 3, 8, 8))
            # Spawn pickling removes parent-owned maps. The worker opens and
            # closes its own maps without mutating the parent backend.
            self.assertEqual(backend.open_shard_count, 0)
            backend.close()


if __name__ == "__main__":
    unittest.main()
