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
    check_hard_negative_readiness,
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
            {"stable_source_id": "A::0::01", "negative_subtype": "wooden_bridge"},
            {"stable_source_id": "B::0::02", "negative_subtype": "flat_floor"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_metadata.parquet"
            fingerprint = write_metadata_sidecar(rows, path)
            sidecar = read_metadata_sidecar(path)
            self.assertEqual(
                sidecar["A::0::01"]["negative_subtype"], "wooden_bridge"
            )
            self.assertEqual(sidecar["A::0::01"]["sample_weight"], 1.0)
            validate_sidecar_against_index(sidecar, videos)
            applied = apply_sidecar(videos, sidecar)
            by_uid = {video.stable_source_id: video for video in applied}
            self.assertEqual(by_uid["A::0::01"].negative_subtype, "wooden_bridge")
            self.assertEqual(by_uid["B::0::02"].negative_subtype, "flat_floor")
            # Untyped videos keep their default.
            self.assertIsNone(by_uid["A::0::02"].negative_subtype)
            # Fingerprint is deterministic and recorded.
            self.assertTrue(fingerprint)
            self.assertEqual(read_metadata_sidecar(path), sidecar)

    def test_fingerprint_changes_with_content(self) -> None:
        rows_a = [{"stable_source_id": "A::01", "negative_subtype": "hard"}]
        rows_b = [{"stable_source_id": "A::01", "negative_subtype": "easy"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            fa = write_metadata_sidecar(rows_a, path)
            fb = write_metadata_sidecar(rows_b, path)
            self.assertNotEqual(fa, fb)

    def test_duplicate_primary_key_is_rejected(self) -> None:
        rows = [
            {"stable_source_id": "A::01", "negative_subtype": "hard"},
            {"stable_source_id": "A::01", "negative_subtype": "easy"},
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "Duplicate metadata"),
        ):
            write_metadata_sidecar(rows, Path(directory) / "meta.parquet")

    def test_tampered_fingerprint_is_rejected(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(
                [{"stable_source_id": "A::01", "negative_subtype": "hard"}],
                path,
            )
            rows = pq.read_table(path).to_pylist()
            rows[0]["negative_subtype"] = "tampered"
            pq.write_table(pa.Table.from_pylist(rows), path)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                read_metadata_sidecar(path)

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
        rows = [{"stable_source_id": "A::0::01", "sample_weight": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(rows, path)
            applied = apply_sidecar(videos, read_metadata_sidecar(path))
            by_uid = {video.stable_source_id: video for video in applied}
            self.assertEqual(by_uid["A::0::01"].sample_weight, 0.5)
            self.assertEqual(by_uid["A::0::02"].sample_weight, 1.0)


class HardNegativeReadinessTests(unittest.TestCase):
    """``read_metadata_sidecar`` treats a missing file as 'no metadata', which
    is right for an optional sidecar and a trap for hard_negative.enabled: the
    hard bucket would be empty and training would silently run plain negative
    sampling. This is the filesystem half of the contract."""

    def _config(self, path, **hard_negative) -> dict:
        base = {
            "enabled": True,
            "subtype_field": "negative_subtype",
            "hard_subtypes": ["wooden_bridge"],
            "ordinary_subtypes": [],
            "negative_mix": {"ordinary": 0.5, "hard": 0.5},
            "min_videos_per_subtype_bucket": 1,
        }
        base.update(hard_negative)
        return {
            "data": {
                "metadata_sidecar": None if path is None else str(path),
                "hard_negative": base,
            }
        }

    def test_disabled_reports_nothing(self) -> None:
        config = self._config("does/not/exist.parquet", enabled=False)
        self.assertEqual(check_hard_negative_readiness(config), [])

    def test_missing_sidecar_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_exists.parquet"
            problems = check_hard_negative_readiness(self._config(path))
            self.assertEqual(len(problems), 1)
            self.assertIn("does not exist", problems[0])
            self.assertIn("silently degrade", problems[0])

    def test_unset_sidecar_is_reported(self) -> None:
        problems = check_hard_negative_readiness(self._config(None))
        self.assertEqual(len(problems), 1)
        self.assertIn("metadata_sidecar", problems[0])

    def test_sidecar_without_hard_subtype_rows_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(
                [
                    {"stable_source_id": "A::01", "negative_subtype": "flat_floor"},
                    {"stable_source_id": "A::02", "negative_subtype": "flat_floor"},
                ],
                path,
            )
            problems = check_hard_negative_readiness(self._config(path))
            self.assertEqual(len(problems), 1)
            self.assertIn("no video whose negative_subtype", problems[0])
            # The message must name what IS there, so the operator can fix the
            # config without opening the parquet.
            self.assertIn("flat_floor", problems[0])
            self.assertIn("sidecar rows: 2", problems[0])

    def test_ready_sidecar_reports_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(
                [
                    {"stable_source_id": "A::01", "negative_subtype": "wooden_bridge"},
                    {"stable_source_id": "A::02", "negative_subtype": "flat_floor"},
                ],
                path,
            )
            self.assertEqual(check_hard_negative_readiness(self._config(path)), [])

    def test_empty_hard_subtypes_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(
                [{"stable_source_id": "A::01", "negative_subtype": "wooden_bridge"}],
                path,
            )
            problems = check_hard_negative_readiness(
                self._config(path, hard_subtypes=[])
            )
            self.assertEqual(len(problems), 1)
            self.assertIn("hard_subtypes", problems[0])

    def test_unpopulated_subtype_field_is_reported(self) -> None:
        # apply_sidecar only ever writes negative_subtype / sample_weight, so
        # any other field reads None for every video and the hard bucket stays
        # empty no matter how well the sidecar is populated.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.parquet"
            write_metadata_sidecar(
                [{"stable_source_id": "A::01", "negative_subtype": "wooden_bridge"}],
                path,
            )
            problems = check_hard_negative_readiness(
                self._config(path, subtype_field="difficulty")
            )
            self.assertEqual(len(problems), 1)
            self.assertIn("difficulty", problems[0])
            self.assertIn("never", problems[0])


if __name__ == "__main__":
    unittest.main()
