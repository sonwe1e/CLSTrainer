"""Packed-shard identity and integrity contract (audit P0-5 / P0-6).

P0-5: ``pack_frame_index`` rebuilt ``VideoEntry`` from the packed frame index,
which carries no per-frame content hash, so ``canonical_stable_source_id``
silently fell back to the ``game::label::video_id`` form. The packed backend
and the PNG backend then disagreed on identity and every sidecar / mining /
subtype join keyed on the uid drifted.

P0-6: ``verify_packed_provenance`` compared ``if current and current !=
recorded``, so a missing or unreadable source index hashed to "" and the
staleness check passed -- absence read as agreement. Separately the
``shard_sha256`` map was written by the packer and never read back, so a
truncated or corrupt shard reached the DataLoader as garbage pixels.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
except ImportError:  # pragma: no cover - environment guard
    np = pa = pq = Image = None

WIDTH = 448
HEIGHT = 208


def _write_frames(root: Path) -> Path:
    """A 4-frame PNG pool over two videos plus its frame index."""
    from game_cls.contract import stamp_parquet_table

    rows = []
    for index in range(4):
        array = np.full((HEIGHT, WIDTH, 3), index * 30, dtype=np.uint8)
        path = root / f"frame_{index}.png"
        Image.fromarray(array).save(path)
        rows.append(
            {
                "path": str(path),
                "game": "game_a",
                "label": index % 2,
                "video_id": f"{index % 2:02d}",
                "frame_id": index,
                "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    frame_index = root / "frames.parquet"
    pq.write_table(stamp_parquet_table(pa.Table.from_pylist(rows)), frame_index)
    return frame_index


def _source_video_index(root: Path) -> Path:
    """A source video index carrying content-anchored canonical uids."""
    from game_cls.data.video_index import (
        VideoEntry,
        video_content_id,
        write_video_entries_parquet,
    )

    frame_index = root / "frames.parquet"
    if not frame_index.is_file():
        _write_frames(root)
    frame_rows = pq.read_table(frame_index).to_pylist()

    entries = []
    for label in (0, 1):
        video_id = f"{label:02d}"
        frame_ids = np.asarray([label, label + 2], dtype=np.int32)
        entries.append(
            VideoEntry(
                game="game_a",
                label=label,
                video_id=video_id,
                frame_ids=frame_ids,
                valid_start_positions={
                    delta: np.asarray([0], dtype=np.int32) for delta in (1, 2, 3)
                },
                stable_source_id=f"game_a::{label}::{video_id}",
                content_version_id=video_content_id(
                    len(frame_ids),
                    [
                        f"{label}:{int(row['frame_id'])}:{row['content_sha256']}"
                        for row in sorted(
                            (
                                item
                                for item in frame_rows
                                if int(item["label"]) == label
                            ),
                            key=lambda item: int(item["frame_id"]),
                        )
                    ],
                )
                or "",
                negative_subtype="bridge" if label == 0 else None,
                sample_weight=2.5 if label == 0 else 1.0,
            )
        )
    path = root / "source_videos.parquet"
    write_video_entries_parquet(entries, path)
    return path


def _pack(root: Path, *, source_video_index: Path | None = None) -> Path:
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.indexing import write_bundle_manifest
    from game_cls.data.packed_backend import pack_frame_index

    frame_index = _write_frames(root)
    source_video_index = source_video_index or _source_video_index(root)
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
    return pack_frame_index(
        frame_index,
        root / "packed",
        image_spec=ImageSpec(width=WIDTH, height=HEIGHT, channels=3),
        images_per_shard=2,
        source_video_index=source_video_index,
        source_bundle_id="bundle-a",
        audit_path=audit,
        split_manifest_path=split_manifest,
    )


@unittest.skipIf(np is None, "packed backend dependencies are not installed")
class PackedIdentityTests(unittest.TestCase):
    """P0-5: the packed index must not invent a different video identity."""

    def test_canonical_uid_is_inherited_from_source_index(self) -> None:
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_video_index(root)
            _pack(root, source_video_index=source)

            packed = read_video_entries_parquet(
                root / "packed" / "packed_video_entries.parquet"
            )
            by_video = {entry.video_id: entry for entry in packed}
            self.assertEqual(
                by_video["00"].stable_source_id,
                "game_a::0::00",
            )
            # Samples and reports use the versioned identity.
            source_entry = read_video_entries_parquet(source)[0]
            self.assertEqual(
                by_video["00"].source_version_id, source_entry.source_version_id
            )

    def test_packed_and_source_uids_agree(self) -> None:
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_video_index(root)
            _pack(root, source_video_index=source)

            source_uids = {
                entry.source_version_id for entry in read_video_entries_parquet(source)
            }
            packed_uids = {
                entry.source_version_id
                for entry in read_video_entries_parquet(
                    root / "packed" / "packed_video_entries.parquet"
                )
            }
            # The whole point of P0-5: these two sets must be identical, or a
            # join keyed on the uid silently drops or mismatches every row.
            self.assertEqual(packed_uids, source_uids)

    def test_sidecar_fields_are_inherited_too(self) -> None:
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_video_index(root)
            _pack(root, source_video_index=source)

            packed = {
                entry.video_id: entry
                for entry in read_video_entries_parquet(
                    root / "packed" / "packed_video_entries.parquet"
                )
            }
            # The same rebuild dropped these, so subtype metrics came back null
            # and per-video sampling weights silently reset to 1.0.
            self.assertEqual(packed["00"].negative_subtype, "bridge")
            self.assertAlmostEqual(packed["00"].sample_weight, 2.5)

    def test_missing_source_index_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_index = _write_frames(root)
            from game_cls.data.image_spec import ImageSpec
            from game_cls.data.packed_backend import pack_frame_index

            with self.assertRaises(FileNotFoundError):
                pack_frame_index(
                    frame_index,
                    root / "packed",
                    image_spec=ImageSpec(width=WIDTH, height=HEIGHT, channels=3),
                    source_video_index=root / "missing.parquet",
                    source_bundle_id="bundle-a",
                )

    def test_frame_id_mismatch_is_refused_without_partial_output(self) -> None:
        from game_cls.data.image_spec import ImageSpec
        from game_cls.data.packed_backend import pack_frame_index
        from game_cls.data.video_index import (
            read_video_entries_parquet,
            write_video_entries_parquet,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_index = _write_frames(root)
            source = _source_video_index(root)
            entries = read_video_entries_parquet(source)
            object.__setattr__(
                entries[0], "frame_ids", np.asarray([99], dtype=np.int32)
            )
            write_video_entries_parquet(entries, source)
            audit = root / "audit.json"
            split_manifest = root / "split_manifest.parquet"
            audit.write_text("{}", encoding="utf-8")
            split_manifest.write_bytes(b"split")
            from game_cls.data.indexing import write_bundle_manifest

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
            output = root / "packed"
            with self.assertRaisesRegex(ValueError, "Frame ids disagree"):
                pack_frame_index(
                    frame_index,
                    output,
                    image_spec=ImageSpec(width=WIDTH, height=HEIGHT, channels=3),
                    source_video_index=source,
                    source_bundle_id="bundle-a",
                    audit_path=audit,
                    split_manifest_path=split_manifest,
                )
            self.assertFalse(output.exists())


@unittest.skipIf(np is None, "packed backend dependencies are not installed")
class PackedProvenanceTests(unittest.TestCase):
    """P0-6: absence must not read as agreement."""

    def test_intact_provenance_passes(self) -> None:
        from game_cls.data.packed_backend import verify_packed_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            verify_packed_provenance(
                root / "packed" / "packed_manifest.json",
                root / "frames.parquet",
                current_video_index=root / "source_videos.parquet",
                audit_path=root / "audit.json",
                split_manifest_path=root / "split_manifest.parquet",
            )
            manifest = json.loads(
                (root / "packed" / "packed_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["contract_version"], 5)
            self.assertEqual(manifest["source_bundle_id"], "bundle-a")
            self.assertTrue(manifest["source_video_index_sha256"])

    def test_changed_bundle_generation_is_refused(self) -> None:
        from game_cls.data.packed_backend import verify_packed_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            payload = json.loads(
                (root / "bundle_manifest.json").read_text(encoding="utf-8")
            )
            payload["bundle_id"] = "bundle-b"
            (root / "bundle_manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                verify_packed_provenance(
                    root / "packed" / "packed_manifest.json",
                    root / "frames.parquet",
                    current_video_index=root / "source_videos.parquet",
                    audit_path=root / "audit.json",
                    split_manifest_path=root / "split_manifest.parquet",
                )

    def test_missing_source_index_is_refused(self) -> None:
        """The core P0-6 defect: a deleted source index used to PASS."""
        from game_cls.data.packed_backend import verify_packed_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            (root / "frames.parquet").unlink()
            with self.assertRaisesRegex(RuntimeError, "cannot be verified"):
                verify_packed_provenance(
                    root / "packed" / "packed_manifest.json",
                    root / "frames.parquet",
                    current_video_index=root / "source_videos.parquet",
                    audit_path=root / "audit.json",
                    split_manifest_path=root / "split_manifest.parquet",
                )

    def test_changed_source_index_is_refused(self) -> None:
        from game_cls.data.packed_backend import verify_packed_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            # Regenerate the index with different content: stale shards.
            from game_cls.contract import stamp_parquet_table

            pq.write_table(
                stamp_parquet_table(
                    pa.Table.from_pylist([{"path": "x", "frame_id": 99}])
                ),
                root / "frames.parquet",
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                verify_packed_provenance(
                    root / "packed" / "packed_manifest.json",
                    root / "frames.parquet",
                    current_video_index=root / "source_videos.parquet",
                    audit_path=root / "audit.json",
                    split_manifest_path=root / "split_manifest.parquet",
                )


@unittest.skipIf(np is None, "packed backend dependencies are not installed")
class PackedShardIntegrityTests(unittest.TestCase):
    """P0-6: shard_sha256 was written but never read back."""

    def test_intact_shards_pass(self) -> None:
        from game_cls.data.packed_backend import verify_packed_shards

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            verify_packed_shards(root / "packed" / "packed_manifest.json")

    def test_corrupt_shard_is_refused(self) -> None:
        from game_cls.data.packed_backend import verify_packed_shards

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            manifest_path = root / "packed" / "packed_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard = root / "packed" / manifest["shards"][0]
            # Flip bytes in place: same length, different pixels. Without the
            # hash check this reaches the model as garbage and shows up only as
            # unexplained loss.
            data = bytearray(shard.read_bytes())
            data[0] = (data[0] + 1) % 256
            shard.write_bytes(bytes(data))
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                verify_packed_shards(manifest_path)

    def test_truncated_shard_is_refused(self) -> None:
        from game_cls.data.packed_backend import verify_packed_shards

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            manifest_path = root / "packed" / "packed_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard = root / "packed" / manifest["shards"][0]
            data = shard.read_bytes()
            shard.write_bytes(data[: len(data) // 2])
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                verify_packed_shards(manifest_path)

    def test_missing_shard_is_refused(self) -> None:
        from game_cls.data.packed_backend import verify_packed_shards

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            manifest_path = root / "packed" / "packed_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            (root / "packed" / manifest["shards"][0]).unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                verify_packed_shards(manifest_path)

    def test_manifest_without_shard_hashes_is_refused(self) -> None:
        """A pre-P0-6 manifest cannot prove its shards are intact."""
        from game_cls.data.packed_backend import verify_packed_shards

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _pack(root)
            manifest_path = root / "packed" / "packed_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["shard_sha256"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no shard_sha256"):
                verify_packed_shards(manifest_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
