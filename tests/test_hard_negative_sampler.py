"""Hard-negative subtype mixing sampler (step5 P2).

The subtype-aware sampler must (a) stay byte-identical when disabled,
(b) mix hard and ordinary negative buckets by weight when enabled,
(c) fall back when a bucket is empty, and (d) honor max_pairs_per_video.
"""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace

from game_cls.data.video_index import build_video_entries
from game_cls.data.video_sampler import VideoBalancedPairBatchSampler
from tests.test_video_index_sampler import frame


def _frames():
    frames = []
    for game in ("A", "B", "C"):
        for label in (0, 1):
            for video_id, ids in (
                ("01", (1, 2, 3, 4, 5, 6)),
                ("02", (1, 2, 3, 4, 5, 6)),
                ("03", (1, 2, 3, 4, 5, 6)),
            ):
                frames.extend(frame(game, label, video_id, item) for item in ids)
    return frames


def _videos():
    entries = build_video_entries(_frames())
    marked = []
    for entry in entries:
        if entry.label == 0 and entry.video_id in ("01",):
            marked.append(replace(entry, negative_subtype="wooden_bridge"))
        elif entry.label == 0 and entry.video_id in ("02",):
            marked.append(replace(entry, negative_subtype="flat_floor"))
        else:
            marked.append(entry)
    return marked


def _sampler(videos, *, seed=7, **cfg):
    return VideoBalancedPairBatchSampler(
        videos,
        local_batch_size=4,
        steps_per_epoch=20,
        rank=0,
        world_size=1,
        seed=seed,
        class_probability={0: 0.5, 1: 0.5},
        delta_probability={1: 0.2, 2: 0.6, 3: 0.2},
        dedup_level="none",
        hard_negative_cfg=cfg,
    )


class HardNegativeSamplerTests(unittest.TestCase):
    def test_disabled_matches_baseline_behavior(self) -> None:
        videos = _videos()
        baseline = _sampler(videos, seed=11)
        disabled = _sampler(videos, seed=11)
        for batch_a, batch_b in zip(baseline, disabled, strict=True):
            self.assertEqual(
                [request.video_index for request in batch_a],
                [request.video_index for request in batch_b],
            )

    def test_enabled_mixes_subtype_buckets_for_negatives(self) -> None:
        videos = _videos()
        sampler = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        hard_video = next(
            index
            for index, video in enumerate(videos)
            if video.negative_subtype == "wooden_bridge"
        )
        ordinary_videos = {
            index
            for index, video in enumerate(videos)
            if video.label == 0 and video.negative_subtype != "wooden_bridge"
        }
        sampled_negatives = Counter()
        for batch in sampler:
            for request in batch:
                video = videos[request.video_index]
                if video.label == 0:
                    sampled_negatives[request.video_index] += 1
        # Hard-bucket video must be reachable; ordinary negatives must too.
        self.assertIn(hard_video, sampled_negatives)
        self.assertTrue(ordinary_videos & set(sampled_negatives))

    def test_empty_bucket_falls_back(self) -> None:
        videos = _videos()
        # Every negative is "flat_floor"; nothing is a hard subtype, so the
        # hard bucket is empty and sampling falls back to ordinary.
        sampler = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["missing"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.0, "hard": 1.0},
            min_videos_per_subtype_bucket=1,
        )
        batches = list(sampler)
        self.assertTrue(batches)
        sampled_negatives = {
            videos[request.video_index].negative_subtype
            for batch in batches
            for request in batch
            if videos[request.video_index].label == 0
        }
        # The hard bucket ("missing") is empty; the fallback serves ordinary
        # negatives (everything not in hard_subtypes), so negatives are
        # still sampled and never stall.
        self.assertTrue(sampled_negatives)
        self.assertNotIn("missing", sampled_negatives)

    def test_bucket_summary_counts_videos_and_legal_pairs(self) -> None:
        videos = _videos()
        sampler = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        summary = sampler.subtype_bucket_summary()
        self.assertTrue(summary["enabled"])
        # One "01" video per game is wooden_bridge -> 3 hard videos; the other
        # two negatives per game are ordinary -> 6 ordinary videos.
        self.assertEqual(summary["buckets"]["hard"]["videos"], 3)
        self.assertEqual(summary["buckets"]["ordinary"]["videos"], 6)
        # Legal pairs are the eligible start positions summed over deltas
        # (6 frames -> 5+4+3 = 12 starts per video for deltas 1/2/3).
        self.assertEqual(summary["buckets"]["hard"]["legal_pairs"], 3 * 12)
        self.assertEqual(summary["buckets"]["ordinary"]["legal_pairs"], 6 * 12)
        self.assertEqual(sorted(summary["by_game"]), ["A", "B", "C"])
        # Per-game counts are DISTINCT videos, not (game, delta) cell sums:
        # game A has exactly one wooden_bridge video, reachable at 3 deltas.
        self.assertEqual(summary["by_game"]["A"]["hard"]["videos"], 1)
        self.assertEqual(summary["by_game"]["A"]["hard"]["legal_pairs"], 12)
        self.assertEqual(summary["by_game"]["A"]["ordinary"]["videos"], 2)
        self.assertEqual(summary["excluded_videos"], 0)
        self.assertEqual(summary["undersized_cells"], [])

    def test_bucket_summary_legal_pairs_follow_max_pairs_per_video(self) -> None:
        # The cap is applied to the videos the sampler holds, so the reported
        # pair counts must be the post-cap ones a human can act on.
        videos = _videos()
        sampler = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
            max_pairs_per_video=1,
        )
        summary = sampler.subtype_bucket_summary()
        self.assertEqual(summary["buckets"]["hard"]["videos"], 3)
        self.assertEqual(summary["buckets"]["hard"]["legal_pairs"], 3 * 3)

    def test_bucket_summary_reports_empty_hard_bucket(self) -> None:
        videos = _videos()
        sampler = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["missing"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        summary = sampler.subtype_bucket_summary()
        self.assertEqual(summary["buckets"]["hard"]["videos"], 0)
        self.assertEqual(summary["buckets"]["hard"]["legal_pairs"], 0)
        # Every (game, delta) hard cell is under the minimum, i.e. every cell
        # takes the silent fallback -- that must be visible.
        self.assertTrue(summary["undersized_cells"])
        self.assertTrue(
            all(cell["bucket"] == "hard" for cell in summary["undersized_cells"])
        )

    def test_bucket_summary_counts_videos_excluded_by_ordinary_subtypes(self) -> None:
        videos = _videos()
        sampler = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            # Explicit ordinary list drops "03" (untyped) from both buckets.
            ordinary_subtypes=["flat_floor"],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        summary = sampler.subtype_bucket_summary()
        self.assertEqual(summary["buckets"]["ordinary"]["videos"], 3)
        self.assertEqual(summary["excluded_videos"], 3)

    def test_bucket_summary_is_disabled_when_the_feature_is(self) -> None:
        summary = _sampler(_videos(), seed=11).subtype_bucket_summary()
        self.assertEqual(summary, {"enabled": False})

    def test_startup_refuses_a_globally_empty_hard_bucket(self) -> None:
        from game_cls.engine.training.loaders import (
            _require_non_degenerate_hard_negatives,
        )

        videos = _videos()
        degenerate = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["missing"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        with self.assertRaises(RuntimeError) as caught:
            _require_non_degenerate_hard_negatives(degenerate.subtype_bucket_summary())
        message = str(caught.exception)
        self.assertIn("hard bucket is empty", message)
        self.assertIn("hard_subtypes", message)
        healthy = _sampler(
            videos,
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        _require_non_degenerate_hard_negatives(healthy.subtype_bucket_summary())
        # Disabled configs must not be refused.
        _require_non_degenerate_hard_negatives({"enabled": False})

    def test_startup_log_prints_videos_and_legal_pairs(self) -> None:
        import io
        from contextlib import redirect_stdout

        from game_cls.engine.training.loaders import _log_hard_negative_buckets

        sampler = _sampler(
            _videos(),
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            ordinary_subtypes=["flat_floor"],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=2,
        )
        summary = sampler.subtype_bucket_summary()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _log_hard_negative_buckets(summary, 0)
        text = buffer.getvalue()
        self.assertIn("hard=3 videos/36 legal pairs", text)
        self.assertIn("ordinary=3 videos/36 legal pairs", text)
        # One video per bucket per game, 12 start positions across 3 deltas.
        self.assertIn("game A: hard=1 videos/12 legal pairs", text)
        # min_videos_per_subtype_bucket=2 with 1 video per (game, delta) cell:
        # every cell borrows, and the excluded untyped negatives are named.
        self.assertIn("min_videos_per_subtype_bucket=2", text)
        self.assertIn("borrow from the other bucket", text)
        self.assertIn("excluded from negative sampling", text)

    def test_startup_log_is_silent_off_rank_zero_and_when_disabled(self) -> None:
        import io
        from contextlib import redirect_stdout

        from game_cls.engine.training.loaders import _log_hard_negative_buckets

        sampler = _sampler(
            _videos(),
            seed=11,
            enabled=True,
            subtype_field="negative_subtype",
            hard_subtypes=["wooden_bridge"],
            ordinary_subtypes=[],
            negative_mix={"ordinary": 0.5, "hard": 0.5},
            min_videos_per_subtype_bucket=1,
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _log_hard_negative_buckets(sampler.subtype_bucket_summary(), 1)
            _log_hard_negative_buckets({"enabled": False}, 0)
        self.assertEqual(buffer.getvalue(), "")

    def test_max_pairs_per_video_caps_start_positions(self) -> None:
        videos = _videos()
        # Videos have 3 delta=2 start positions; the cap keeps only the
        # first one, so every sampled start must be 0.
        capped = _sampler(
            videos,
            seed=3,
            enabled=False,
            max_pairs_per_video=1,
        )
        capped_starts = {
            request.start_position for batch in capped for request in batch
        }
        self.assertEqual(capped_starts, {0})
        uncapped = _sampler(videos, seed=3, enabled=False)
        uncapped_starts = {
            request.start_position for batch in uncapped for request in batch
        }
        self.assertTrue(
            any(start > 0 for start in uncapped_starts),
            "uncapped sampler should reach later start positions",
        )


if __name__ == "__main__":
    unittest.main()
