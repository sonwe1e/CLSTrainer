"""Source-video-level deterministic splitter tests (step4 §二)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from game_cls.data.records import FrameRecord
from game_cls.data.splitter import (
    SPLIT_ALGORITHM_VERSION,
    compute_dataset_fingerprint,
    extend_split,
    fingerprint_covers_content,
    load_split_manifest,
    resolve_split,
    source_video_uid,
    split_source_videos,
    split_summary,
    write_split_manifest,
)


def _frames(
    game: str,
    label: int,
    video_id: str,
    count: int,
    split="train",
    root: str = "/data",
    content_hash: bool = True,
):
    return [
        FrameRecord(
            sample_id=f"{split}:{game}:{label}:{video_id}:{i:05d}",
            split=split,
            game=game,
            label=label,
            video_id=video_id,
            frame_id=i,
            path=f"{root}/{game}/{label}/{video_id}{i:05d}.png",
            width=448,
            height=208,
            channels=3,
            file_size=1,
            content_sha256=f"sha-{game}-{label}-{video_id}-{i}" if content_hash else "",
        )
        for i in range(count)
    ]


def _dataset(
    *,
    videos_per_game_label: int = 4,
    frames_per_video: int = 12,
    root: str = "/data",
    content_hash: bool = True,
):
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
                        root=root,
                        content_hash=content_hash,
                    )
                )
    return frames


def _ragged_dataset(frame_counts: tuple[int, ...]):
    """Same layout as ``_dataset`` but with uneven per-video frame counts.

    Uniform runs give every delta the same pair ratio, which hides a
    delta-specific reporting bug; uneven runs separate them.
    """
    frames: list[FrameRecord] = []
    for game_index, game in enumerate(("game_a", "game_b", "game_c")):
        for label in (0, 1):
            for video, count in enumerate(frame_counts):
                frames.extend(
                    _frames(game, label, f"{game_index:02d}{label}{video:02d}", count)
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

    def test_fingerprint_ignores_the_dataset_root(self) -> None:
        # Same dataset staged under a different mount point must keep its
        # fingerprint, otherwise a valid manifest is refused after a move.
        here = _dataset(root="/data")
        moved = _dataset(root="/mnt/nfs/other/place")
        self.assertNotEqual([f.path for f in here], [f.path for f in moved])
        self.assertEqual(
            compute_dataset_fingerprint(here),
            compute_dataset_fingerprint(moved),
        )

    def test_fingerprint_tracks_edited_content(self) -> None:
        frames = _dataset()
        edited = list(frames)
        # Same name, same size, different bytes: only the content hash
        # distinguishes it.
        edited[0] = FrameRecord(
            **{**edited[0].to_dict(), "content_sha256": "sha-edited"}
        )
        self.assertNotEqual(
            compute_dataset_fingerprint(edited),
            compute_dataset_fingerprint(frames),
        )

    def test_fingerprint_content_coverage_is_reported(self) -> None:
        self.assertTrue(fingerprint_covers_content(_dataset()))
        self.assertFalse(fingerprint_covers_content(_dataset(content_hash=False)))

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
        assignment = split_source_videos(frames, val_ratio=0.2, seed=7, target_delta=2)
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

    def test_summary_reports_the_balancing_delta(self) -> None:
        # Ragged frame counts make the three per-delta ratios differ, so
        # "achieved == delta2" cannot pass by coincidence.
        frames = _ragged_dataset((4, 5, 30, 31))
        for delta in (1, 2, 3):
            with self.subTest(target_delta=delta):
                assignment = split_source_videos(
                    frames, val_ratio=0.25, seed=13, target_delta=delta
                )
                summary = split_summary(
                    frames,
                    assignment,
                    dataset_fingerprint="fp",
                    seed=13,
                    val_ratio=0.25,
                    target_delta=delta,
                )
                per_delta = [summary[f"val_ratio_achieved_delta{d}"] for d in (1, 2, 3)]
                self.assertEqual(
                    len(set(per_delta)), 3, "fixture must separate the deltas"
                )
                self.assertEqual(summary["target_delta"], delta)
                # val_ratio_achieved follows target_delta instead of always
                # reporting the delta=2 number.
                self.assertEqual(
                    summary["val_ratio_achieved"],
                    summary[f"val_ratio_achieved_delta{delta}"],
                )

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
        self.assertEqual(assignment[source_video_uid("only_game", "01")], "train")


class SplitManifestTests(unittest.TestCase):
    def test_write_load_roundtrip(self) -> None:
        frames = _dataset()
        assignment = split_source_videos(frames, val_ratio=0.2, seed=5, target_delta=2)
        fingerprint = compute_dataset_fingerprint(frames)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            write_split_manifest(
                frames,
                assignment,
                manifest,
                dataset_fingerprint=fingerprint,
                seed=5,
                val_ratio=0.2,
                target_delta=2,
            )
            loaded = load_split_manifest(manifest)
        self.assertEqual(loaded["assignment"], assignment)
        self.assertEqual(loaded["dataset_fingerprint"], fingerprint)
        self.assertEqual(loaded["split_seed"], 5)
        self.assertEqual(loaded["split_algorithm_version"], SPLIT_ALGORITHM_VERSION)
        # The balancing parameters ride along so reuse can verify them.
        self.assertAlmostEqual(loaded["split_val_ratio"], 0.2)
        self.assertEqual(loaded["split_target_delta"], 2)

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
        assignment = split_source_videos(frames, val_ratio=0.2, seed=3, target_delta=2)
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

    def test_resolve_rejects_a_changed_seed(self) -> None:
        self._assert_parameter_change_rejected(seed=10)

    def test_resolve_rejects_a_changed_val_ratio(self) -> None:
        self._assert_parameter_change_rejected(val_ratio=0.3)

    def test_resolve_rejects_a_changed_target_delta(self) -> None:
        self._assert_parameter_change_rejected(target_delta=3)

    def _assert_parameter_change_rejected(self, **changed) -> None:
        """Same data, different split policy -> refuse to reuse or extend."""
        frames = _dataset()
        params = {"val_ratio": 0.2, "seed": 9, "target_delta": 2}
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            resolve_split(frames, manifest_path=manifest, **params)
            for policy in ("error", "extend"):
                # Even extend must refuse: the manifest was balanced under a
                # different contract, so mixing them is not what was asked.
                with self.assertRaises(ValueError) as caught:
                    resolve_split(
                        frames,
                        manifest_path=manifest,
                        on_new_groups=policy,
                        **{**params, **changed},
                    )
                message = str(caught.exception)
                self.assertIn("different split parameters", message)
                for field in changed:
                    self.assertIn(
                        field.replace("val_ratio", "split_val_ratio"), message
                    )

    def test_resolve_rejects_a_stale_algorithm_version(self) -> None:
        frames = _dataset()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            resolve_split(
                frames,
                val_ratio=0.2,
                seed=9,
                target_delta=2,
                manifest_path=manifest,
            )
            # Rewrite the manifest as if an older algorithm version produced
            # it: reuse must refuse rather than trust a stale rule.
            from game_cls.data.indexing import _pyarrow

            pa, pq = _pyarrow()
            rows = pq.read_table(manifest).to_pylist()
            for row in rows:
                row["split_algorithm_version"] = SPLIT_ALGORITHM_VERSION - 1
                row.pop("split_val_ratio")
                row.pop("split_target_delta")
            pq.write_table(pa.Table.from_pylist(rows), manifest)
            with self.assertRaises(ValueError) as caught:
                resolve_split(
                    frames,
                    val_ratio=0.2,
                    seed=9,
                    target_delta=2,
                    manifest_path=manifest,
                )
        self.assertIn("split_algorithm_version", str(caught.exception))

    def test_load_split_manifest_releases_file_handle(self) -> None:
        """load_split_manifest must not retain a pyarrow file handle.

        pq.ParquetFile opens without FILE_SHARE_DELETE on Windows, so any
        live reference blocks os.replace on the same path with WinError 32
        (ERROR_SHARING_VIOLATION).  resolve_split atomically rewrites a
        stale manifest; if load_split_manifest leaked a handle that rewrite
        would fail silently on Windows but pass on Linux CI — exactly the
        platform-specific silent failure this review effort targets.

        The returned dict is kept in scope so that a future implementation
        accidentally embedding a ParquetFile or Table in the return value
        would be caught here rather than only in TemporaryDirectory cleanup.
        """
        frames = _dataset()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            resolve_split(frames, val_ratio=0.2, seed=5, manifest_path=manifest)

            # Hold the mapping alive across the replacement — any pyarrow
            # object embedded in it would keep its file handle open here.
            loaded = load_split_manifest(manifest)

            # Atomic replacement must succeed while `loaded` is in scope.
            # PermissionError (WinError 32) fires if the file is held open
            # without FILE_SHARE_DELETE.
            replacement = Path(directory) / "_replacement.parquet"
            replacement.write_bytes(manifest.read_bytes())
            replacement.replace(manifest)  # must not raise

            self.assertIn("assignment", loaded)
            self.assertIsNotNone(loaded["dataset_fingerprint"])

    def test_resolve_rejects_an_unknown_on_new_groups(self) -> None:
        with self.assertRaises(ValueError):
            resolve_split(
                _dataset(),
                val_ratio=0.2,
                seed=1,
                on_new_groups="reshuffle",
            )


class ExtendSplitTests(unittest.TestCase):
    def test_extend_keeps_every_existing_assignment(self) -> None:
        frames = _dataset()
        base = split_source_videos(frames, val_ratio=0.2, seed=9, target_delta=2)
        grown = frames + _frames("game_a", 1, "0199", 12)
        assignment, stats = extend_split(
            grown,
            base,
            val_ratio=0.2,
            seed=9,
            target_delta=2,
        )
        for uid, split in base.items():
            self.assertEqual(assignment[uid], split, f"{uid} moved")
        new_uid = source_video_uid("game_a", "0199")
        self.assertIn(new_uid, assignment)
        self.assertEqual(stats["added_source_videos"], 1)
        self.assertEqual(stats["added_source_video_uids"], [new_uid])
        self.assertEqual(stats["dropped_source_videos"], 0)

    def test_extend_drops_source_videos_that_disappeared(self) -> None:
        frames = _dataset()
        base = split_source_videos(frames, val_ratio=0.2, seed=9, target_delta=2)
        base["game_a::deleted"] = "val"
        assignment, stats = extend_split(
            frames,
            base,
            val_ratio=0.2,
            seed=9,
            target_delta=2,
        )
        self.assertNotIn("game_a::deleted", assignment)
        self.assertEqual(stats["dropped_source_videos"], 1)
        self.assertEqual(stats["dropped_source_video_uids"], ["game_a::deleted"])

    def test_extend_is_deterministic(self) -> None:
        frames = _dataset()
        base = split_source_videos(frames, val_ratio=0.2, seed=9, target_delta=2)
        grown = frames + _frames("game_b", 0, "0198", 12)
        first, _ = extend_split(grown, base, val_ratio=0.2, seed=9)
        second, _ = extend_split(grown, base, val_ratio=0.2, seed=9)
        self.assertEqual(first, second)

    def test_extend_rejects_a_new_unsplittable_stratum(self) -> None:
        frames = _dataset()
        base = split_source_videos(frames, val_ratio=0.2, seed=9, target_delta=2)
        grown = frames + _frames("brand_new_game", 1, "01", 12)
        with self.assertRaises(ValueError):
            extend_split(grown, base, val_ratio=0.2, seed=9)
        assignment, _ = extend_split(
            grown,
            base,
            val_ratio=0.2,
            seed=9,
            small_stratum_policy="warn",
        )
        # warn keeps the lone new video out of validation.
        self.assertEqual(assignment[source_video_uid("brand_new_game", "01")], "train")

    def test_extend_keeps_the_ratio_near_target(self) -> None:
        frames = _dataset(videos_per_game_label=10, frames_per_video=20)
        base = split_source_videos(frames, val_ratio=0.2, seed=9, target_delta=2)
        grown = list(frames)
        for game_index, game in enumerate(("game_a", "game_b", "game_c")):
            for label in (0, 1):
                for video in range(10, 20):
                    grown.extend(
                        _frames(game, label, f"{game_index:02d}{label}{video:02d}", 20)
                    )
        assignment, stats = extend_split(
            grown,
            base,
            val_ratio=0.2,
            seed=9,
            target_delta=2,
        )
        self.assertEqual(stats["added_source_videos"], 60)
        summary = split_summary(
            grown,
            assignment,
            dataset_fingerprint="fp",
            seed=9,
            val_ratio=0.2,
            target_delta=2,
        )
        # The greedy walk resumes from the pairs val already holds, so the
        # union converges on the target instead of stacking two 20% halves.
        self.assertLess(abs(summary["val_ratio_achieved"] - 0.2), 0.05)

    def test_resolve_extends_and_rewrites_the_manifest(self) -> None:
        frames = _dataset()
        grown = frames + _frames("game_c", 0, "0197", 12)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            base, _ = resolve_split(
                frames,
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
            )
            assignment, summary = resolve_split(
                grown,
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
                on_new_groups="extend",
            )
            self.assertTrue(summary["manifest_extended"])
            self.assertFalse(summary["manifest_reused"])
            self.assertEqual(summary["added_source_videos"], 1)
            for uid, split in base.items():
                self.assertEqual(assignment[uid], split)

            # The rewritten manifest carries the new fingerprint, so the next
            # run reuses it instead of extending the stale base again.
            again, reused_summary = resolve_split(
                grown,
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
                on_new_groups="error",
            )
            self.assertTrue(reused_summary["manifest_reused"])
            self.assertFalse(reused_summary["manifest_extended"])
            self.assertEqual(again, assignment)

    def test_resolve_reports_fingerprint_content_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "split_manifest.parquet"
            _, summary = resolve_split(
                _dataset(content_hash=False),
                val_ratio=0.2,
                seed=9,
                manifest_path=manifest,
            )
        self.assertFalse(summary["fingerprint_covers_content"])


if __name__ == "__main__":
    unittest.main()
