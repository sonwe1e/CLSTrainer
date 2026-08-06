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
    def test_disabled_matches_legacy_behavior(self) -> None:
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
