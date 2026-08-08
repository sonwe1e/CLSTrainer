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
    def _contract(self, root: Path, rows: list[dict]) -> dict:
        import hashlib

        from game_cls.contract import stamp_parquet_table
        from game_cls.data.indexing import write_bundle_manifest
        from game_cls.data.video_index import (
            VideoEntry,
            video_content_id,
            write_video_entries_parquet,
        )

        for row in rows:
            row["content_sha256"] = hashlib.sha256(
                Path(row["path"]).read_bytes()
            ).hexdigest()
        pq.write_table(
            stamp_parquet_table(pa.Table.from_pylist(rows)), root / "frames.parquet"
        )

        grouped = {}
        for row in rows:
            key = (row["game"], int(row["label"]), row["video_id"])
            grouped.setdefault(key, []).append(int(row["frame_id"]))
        entries = []
        for (game, label, video_id), frame_ids_list in grouped.items():
            frame_ids = np.asarray(sorted(frame_ids_list), dtype=np.int32)
            id_set = set(frame_ids.tolist())
            entries.append(
                VideoEntry(
                    game=game,
                    label=label,
                    video_id=video_id,
                    frame_ids=frame_ids,
                    valid_start_positions={
                        delta: np.asarray(
                            [
                                i
                                for i, value in enumerate(frame_ids)
                                if int(value) + delta in id_set
                            ],
                            dtype=np.int32,
                        )
                        for delta in (1, 2, 3)
                    },
                    stable_source_id=f"{game}::{label}::{video_id}",
                    content_version_id=video_content_id(
                        len(frame_ids),
                        [
                            f"{label}:{int(row['frame_id'])}:{row['content_sha256']}"
                            for row in sorted(
                                (
                                    item
                                    for item in rows
                                    if item["game"] == game
                                    and int(item["label"]) == label
                                    and item["video_id"] == video_id
                                ),
                                key=lambda item: int(item["frame_id"]),
                            )
                        ],
                    )
                    or "",
                )
            )
        video_index = root / "source_videos.parquet"
        write_video_entries_parquet(entries, video_index)
        audit = root / "audit.json"
        split_manifest = root / "split_manifest.parquet"
        audit.write_text("{}", encoding="utf-8")
        split_manifest.write_bytes(b"split")
        write_bundle_manifest(
            root,
            "bundle-a",
            artifact_names=(
                "frames.parquet",
                "source_videos.parquet",
                "audit.json",
                "split_manifest.parquet",
            ),
        )
        return {
            "source_video_index": video_index,
            "source_bundle_id": "bundle-a",
            "audit_path": audit,
            "split_manifest_path": split_manifest,
        }

    def test_packed_decode_matches_png_pixels(self) -> None:
        from game_cls.contract import stamp_parquet_table
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
            pq.write_table(stamp_parquet_table(pa.Table.from_pylist(rows)), frame_index)
            image_spec = ImageSpec(width=448, height=208, channels=3)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                image_spec=image_spec,
                images_per_shard=2,
                **self._contract(root, rows),
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
            # Audit P0-6: the manifest binds the shards to their source.
            self.assertIn("source_frame_index_sha256", manifest)
            self.assertIn("shard_sha256", manifest)
            with self.assertRaisesRegex(ValueError, "does not match configured shape"):
                PackedUint8Backend(
                    packed_index,
                    image_spec=ImageSpec(width=208, height=448, channels=3),
                )
            backend.close()

    def test_stale_source_index_is_refused(self) -> None:
        # Audit P0-6 / acceptance #4: modify the frame index after packing and
        # the old shards must be refused before a DataLoader is built.
        from game_cls.contract import stamp_parquet_table
        from game_cls.data.image_spec import ImageSpec
        from game_cls.data.packed_backend import (
            pack_frame_index,
            verify_packed_provenance,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(4):
                array = np.full((208, 448, 3), index * 40, dtype=np.uint8)
                path = root / f"image_{index}.png"
                Image.fromarray(array).save(path)
                rows.append(
                    {
                        "path": str(path),
                        "game": "g",
                        "label": 0,
                        "video_id": "01",
                        "frame_id": index,
                    }
                )
            frame_index = root / "frames.parquet"
            pq.write_table(stamp_parquet_table(pa.Table.from_pylist(rows)), frame_index)
            image_spec = ImageSpec(width=448, height=208, channels=3)
            contract = self._contract(root, rows)
            pack_frame_index(
                frame_index,
                root / "packed",
                image_spec=image_spec,
                **contract,
            )
            manifest = root / "packed" / "packed_manifest.json"
            # Unchanged source index verifies cleanly.
            verify_packed_provenance(
                manifest,
                frame_index,
                current_video_index=contract["source_video_index"],
                audit_path=contract["audit_path"],
                split_manifest_path=contract["split_manifest_path"],
            )
            # A regenerated (different) source index -> stale -> refused.
            pq.write_table(
                stamp_parquet_table(
                    pa.Table.from_pylist(
                        [
                            dict(row, frame_id=index + 100)
                            for index, row in enumerate(rows)
                        ]
                    )
                ),
                frame_index,
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                verify_packed_provenance(
                    manifest,
                    frame_index,
                    current_video_index=contract["source_video_index"],
                    audit_path=contract["audit_path"],
                    split_manifest_path=contract["split_manifest_path"],
                )

    def test_manifest_without_provenance_is_refused(self) -> None:
        # A manifest that predates the provenance binding cannot prove the
        # shards match the current index; demand a repack instead of silently
        # reading possibly-stale pixels.
        import json as _json

        from game_cls.data.packed_backend import verify_packed_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "packed_manifest.json"
            manifest.write_text(_json.dumps({"shards": []}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "contract_version=5"):
                verify_packed_provenance(manifest, root / "frames.parquet")

    def test_packed_dataset_spawn_worker_owns_its_memmap(self) -> None:
        from torch.utils.data import DataLoader

        from game_cls.contract import stamp_parquet_table
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
            pq.write_table(stamp_parquet_table(pa.Table.from_pylist(rows)), frame_index)
            image_spec = ImageSpec(width=8, height=8, channels=3)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                image_spec=image_spec,
                images_per_shard=2,
                **self._contract(root, rows),
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
