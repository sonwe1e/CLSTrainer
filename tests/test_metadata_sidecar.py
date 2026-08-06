"""Per-video metadata sidecar (step5 P2).

Covers the additive sidecar protocol: write -> read round-trip, uid
validation against the index, post-split application to VideoEntry, and
fingerprint change detection.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from game_cls.data.sidecar import (
    apply_sidecar,
    read_metadata_sidecar,
    validate_sidecar_against_index,
    write_metadata_sidecar,
)
from game_cls.data.video_index import build_video_entries
from tests.test_video_index_sampler import frame


def _videos():
    frames = []
    for game in ("A", "B"):
        for label in (0, 1):
            for video_id, ids in (("01", (1, 2, 3, 4)), ("02", (1, 2, 3))):
                frames.extend(frame(game, label, video_id, item) for item in ids)
    return build_video_entries(frames)


class MetadataSidecarTests(unittest.TestCase):
    def test_write_read_apply_roundtrip(self) -> None:
        videos = _videos()
        rows = [
            {"source_video_uid": "A::01", "negative_subtype": "wooden_bridge"},
            {"source_video_uid": "B::02", "negative_subtype": "flat_floor"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_metadata.parquet"
            fingerprint = write_metadata_sidecar(rows, path)
            sidecar = read_metadata_sidecar(path)
            self.assertEqual(sidecar["A::01"]["negative_subtype"], "wooden_bridge")
            self.assertEqual(sidecar["A::01"]["sample_weight"], 1.0)
            validate_sidecar_against_index(sidecar, videos)
            applied = apply_sidecar(videos, sidecar)
            by_uid = {video.source_video_uid: video for video in applied}
            self.assertEqual(by_uid["A::01"].negative_subtype, "wooden_bridge")
            self.assertEqual(by_uid["B::02"].negative_subtype, "flat_floor")
            # Untyped videos keep their default.
            self.assertIsNone(by_uid["A::02"].negative_subtype)
            # Fingerprint is deterministic and recorded.
            self.assertTrue(fingerprint)
            self.assertEqual(read_metadata_sidecar(path), sidecar)

    def test_fingerprint_changes_with_content(self) -> None:
        rows_a = [{"source_video_uid": "A::01", "negative_subtype": "hard"}]
        rows_b = [{"source_video_uid": "A::01", "negative_subtype": "easy"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            fa = write_metadata_sidecar(rows_a, path)
            fb = write_metadata_sidecar(rows_b, path)
            self.assertNotEqual(fa, fb)

    def test_unknown_uid_rejected(self) -> None:
        videos = _videos()
        sidecar = {"missing::99": {"negative_subtype": "wooden_bridge"}}
        with self.assertRaises(ValueError):
            validate_sidecar_against_index(sidecar, videos)

    def test_missing_sidecar_reads_as_empty(self) -> None:
        self.assertEqual(read_metadata_sidecar("does/not/exist.parquet"), {})

    def test_apply_without_sidecar_is_identity(self) -> None:
        videos = _videos()
        self.assertIs(apply_sidecar(videos, {}), videos)

    def test_sample_weight_applied(self) -> None:
        videos = _videos()
        rows = [{"source_video_uid": "A::01", "sample_weight": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(rows, path)
            applied = apply_sidecar(videos, read_metadata_sidecar(path))
            by_uid = {video.source_video_uid: video for video in applied}
            self.assertEqual(by_uid["A::01"].sample_weight, 0.5)
            self.assertEqual(by_uid["A::02"].sample_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
