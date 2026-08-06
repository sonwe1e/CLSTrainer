"""Source-video-level deterministic splitter tests (step4 §二)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from game_cls.data.records import FrameRecord
from game_cls.data.splitter import (
    SPLIT_ALGORITHM_VERSION,
    compute_dataset_fingerprint,
    load_split_manifest,
    resolve_split,
    source_video_uid,
    split_source_videos,
    split_summary,
    write_split_manifest,
)


def _frames(game: str, label: int, video_id: str, count: int, split="train"):
    return [
        FrameRecord(
            sample_id=f"{split}:{game}:{label}:{video_id}:{i:05d}",
            split=split,
            game=game,
            label=label,
            video_id=video_id,
            frame_id=i,
            path=f"/data/{game}/{label}/{video_id}{i:05d}.png",
            width=448,
            height=208,
            channels=3,
            file_size=1,
        )
        for i in range(count)
    ]


def _dataset(*, videos_per_game_label: int = 4, frames_per_video: int = 12):
    """Three games x two labels x N videos, each a clean run of frames."""
    frames: list[FrameRecord] = []
    for game_index, game in enumerate(("game_a", "game_b", "game_c")):
        for label in (0, 1):
            for video in range(videos_per_game_label):
                frames.extend(
                    _frames(
                        game,
                        label,
                        f"{game_index:02d}{label}{video:02d}",
                        frames_per_video,
                    )
                )
    return frames


class SplitterCoreTests(unittest.TestCase):
    def test_source_video_uid_is_label_independent(self) -> None:
        self.assertEqual(source_video_uid("g", "01"), "g::01")
        self.assertNotEqual(source_video_uid("g", "01"), source_video_uid("h", "01"))

    def test_fingerprint_is_deterministic_and_sensitive(self) -> None:
        frames = _dataset()
        self.assertEqual(
            compute_dataset_fingerprint(frames),
            compute_dataset_fingerprint(list(reversed(frames))),
        )
        changed = list(frames)
        changed[0] = FrameRecord(**{**changed[0].to_dict(), "frame_id": 999})
        self.assertNotEqual(
            compute_dataset_fingerprint(changed),
            compute_dataset_fingerprint(frames),
        )

    def test_no_source_video_spans_splits(self) -> None:
        frames = _dataset()
        assignment = split_source_videos(
            frames, val_ratio=0.2, seed=20260728, target_delta=2
        )
        uids = {source_video_uid(f.game, f.video_id) for f in frames}
        self.assertEqual(set(assignment), uids)
        # Every source video maps to exactly one split, so no overlap.
        train = {
            source_video_uid(f.game, f.video_id)
            for f in frames
            if assignment[source_video_uid(f.game, f.video_id)] == "train"
        }
        val = {
            source_video_uid(f.game, f.video_id)
            for f in frames
            if assignment[source_video_uid(f.game, f.video_id)] == "val"
        }
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertEqual(train & val, set())

    def test_deterministic_across_calls(self) -> None:
        frames = _dataset()
        a = split_source_videos(frames, val_ratio=0.2, seed=42)
        b = split_source_videos(frames, val_ratio=0.2, seed=42)
        self.assertEqual(a, b)

    def test_pair_ratio_approaches_target(self) -> None:
        frames = _dataset(videos_per_game_label=12, frames_per_video=20)
        assignment = split_source_videos(
            frames, val_ratio=0.2, seed=7, target_delta=2
        )
        summary = split_summary(
            frames,
            assignment,
            dataset_fingerprint=compute_dataset_fingerprint(frames),
            seed=7,
            val_ratio=0.2,
            target_delta=2,
        )
        achieved = summary["val_ratio_achieved_delta2"]
        # Greedy grouping over many source videos should be within 5 points.
        self.assertLess(abs(achieved - 0.2), 0.05)
        self.assertGreater(summary["splits"]["val"]["source_video_count"], 0)
        self.assertGreater(summary["splits"]["train"]["source_video_count"], 0)

    def test_small_stratum_policy_error(self) -> None:
        # One game, one label, one video -> cannot split without leakage.
        frames = _frames("only_game", 1, "01", 20)
        with self.assertRaises(ValueError):
            split_source_videos(frames, val_ratio=0.2, seed=1)

    def test_small_stratum_policy_warn_keeps_train(self) -> None:
        frames = _frames("only_game", 1, "01", 20)
        assignment = split_source_videos(
            frames,
            val_ratio=0.2,
            seed=1,
            small_stratum_policy="warn",
        )
        self.assertEqual(
            assignment[source_video_uid("only_game", "01")], "train"
        )


class SplitManifestTests(unittest.TestCase):
    def test_write_load_roundtrip(self) -> None:
        frames = _dataset()
        assignment = split_source_videos(
            frames, val_ratio=0.2, seed=5, target_delta=2
        )
        fingerprint = compute_dataset_fingerprint(frames)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            write_split_manifest(
                frames,
                assignment,
                manifest,
                dataset_fingerprint=fingerprint,
                seed=5,
            )
            loaded = load_split_manifest(manifest)
        self.assertEqual(loaded["assignment"], assignment)
        self.assertEqual(loaded["dataset_fingerprint"], fingerprint)
        self.assertEqual(loaded["split_seed"], 5)
        self.assertEqual(
            loaded["split_algorithm_version"], SPLIT_ALGORITHM_VERSION
        )

    def test_resolve_reuses_matching_manifest(self) -> None:
        frames = _dataset()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            first, _ = resolve_split(
                frames,
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
                on_new_groups="error",
            )
            second, summary = resolve_split(
                frames,
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
                on_new_groups="error",
            )
            self.assertEqual(first, second)
            self.assertTrue(summary["manifest_reused"])

    def test_resolve_rejects_changed_fingerprint(self) -> None:
        frames = _dataset()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            resolve_split(
                frames,
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
                on_new_groups="error",
            )
            added = frames + _frames("new_game", 1, "01", 10)
            with self.assertRaises(ValueError):
                resolve_split(
                    added,
                    val_ratio=0.2,
                    seed=9,
                    manifest_path=manifest,
                    on_new_groups="error",
                )

    def test_split_summary_shape(self) -> None:
        frames = _dataset()
        assignment = split_source_videos(
            frames, val_ratio=0.2, seed=3, target_delta=2
        )
        summary = split_summary(
            frames,
            assignment,
            dataset_fingerprint="fp",
            seed=3,
            val_ratio=0.2,
            target_delta=2,
        )
        self.assertEqual(summary["split_algorithm_version"], SPLIT_ALGORITHM_VERSION)
        self.assertIn("train", summary["splits"])
        self.assertIn("val", summary["splits"])
        self.assertEqual(summary["source_video_count"], len(assignment))


if __name__ == "__main__":
    unittest.main()
