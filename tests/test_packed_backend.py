from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

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
        from game_cls.data.packed_backend import (
            PackedUint8Backend,
            pack_frame_index,
        )
        from game_cls.data.lazy_pair_dataset import build_eval_dataset
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            expected = []
            for index in range(5):
                array = np.full((8, 6, 3), index * 40, dtype=np.uint8)
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
                expected.append(torch.from_numpy(
                    array.transpose(2, 0, 1).copy()
                ))
            frame_index = root / "frames.parquet"
            pq.write_table(pa.Table.from_pylist(rows), frame_index)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                images_per_shard=2,
                expected_width=6,
                expected_height=8,
            )
            backend = PackedUint8Backend(
                packed_index,
                channels=3,
                height=8,
                width=6,
                max_open_shards=1,
            )
            for location, tensor in enumerate(expected):
                self.assertTrue(torch.equal(backend(location), tensor))
                self.assertLessEqual(backend.open_shard_count, 1)
            self.assertEqual(
                len(list((root / "packed").glob("shard_*.bin"))), 3
            )
            self.assertTrue(
                (root / "packed" / "packed_video_entries.parquet").is_file()
            )
            videos = read_video_entries_parquet(
                root / "packed" / "packed_video_entries.parquet",
                deltas=(2,),
            )
            dataset = build_eval_dataset(videos, 2, decoder=backend)
            sample = dataset[0]
            self.assertEqual(tuple(sample["images"].shape), (2, 3, 8, 6))
            self.assertTrue(
                sample["meta"]["image0_path"].startswith("packed://frame/")
            )
            manifest = (
                root / "packed" / "packed_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str((root / "packed").resolve()), manifest)
            backend.close()


if __name__ == "__main__":
    unittest.main()
