"""Sampling deduplication semantics (step3 高优-4)."""

from __future__ import annotations

import random
import unittest

from game_cls.data.pair_sampler import _weighted_choice
from game_cls.data.video_index import VideoEntry
from game_cls.data.video_sampler import VideoBalancedPairBatchSampler, _choice


def _videos(count: int = 8, starts: int = 3) -> list[VideoEntry]:
    import numpy as np

    entries = []
    for index in range(count):
        entries.append(
            VideoEntry(
                game=f"game_{index % 2}",
                video_id=f"{index:02d}",
                label=index % 2,
                frame_ids=np.arange(starts + 2, dtype=np.int64) + index * 100,
                valid_start_positions={
                    delta: np.arange(starts, dtype=np.int64) for delta in (1, 2, 3)
                },
            )
        )
    return entries


class ChoiceWeightTests(unittest.TestCase):
    def test_all_zero_weights_raise_instead_of_uniform_fallback(self) -> None:
        rng = random.Random(0)
        with self.assertRaisesRegex(ValueError, "weights are <= 0"):
            _choice(rng, ["a", "b"], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "weights are <= 0"):
            _weighted_choice(rng, ["a", "b"], [-1.0, 0.0])

    def test_positive_weights_still_work(self) -> None:
        rng = random.Random(0)
        self.assertIn(_choice(rng, ["a", "b"], [1.0, 0.0]), ("a", "b"))


class DedupLevelTests(unittest.TestCase):
    def test_video_level_dedup_never_repeats_a_video_in_a_batch(self) -> None:
        entries = _videos(count=8, starts=3)
        sampler = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=8,
            steps_per_epoch=10,
            seed=3,
            dedup_level="video",
        )
        for batch in sampler:
            video_indices = [request.video_index for request in batch]
            self.assertEqual(len(video_indices), len(set(video_indices)))

    def test_pair_level_dedup_allows_same_video_other_starts(self) -> None:
        entries = _videos(count=8, starts=3)
        sampler = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=8,
            steps_per_epoch=10,
            seed=3,
            dedup_level="pair",
        )
        same_video_in_batch = False
        for batch in sampler:
            identities = {(r.video_index, r.delta, r.start_position) for r in batch}
            # Within one batch every pair identity is unique...
            self.assertEqual(len(identities), len(batch))
            # ...but the same video may legitimately appear again with a
            # different (delta, start): pair-level dedup is weaker than
            # video-level.
            video_counts = [r.video_index for r in batch]
            if len(video_counts) != len(set(video_counts)):
                same_video_in_batch = True
        self.assertTrue(same_video_in_batch)

    def test_exhaustion_error_raises_when_pool_too_small(self) -> None:
        entries = _videos(count=1, starts=1)
        sampler = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=4,
            steps_per_epoch=1,
            seed=3,
            dedup_level="video",
            on_exhaustion="error",
        )
        with self.assertRaisesRegex(RuntimeError, "Deduplication exhausted"):
            list(sampler)

    def test_exhaustion_warn_and_relax_accepts_duplicates(self) -> None:
        entries = _videos(count=1, starts=1)
        sampler = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=4,
            steps_per_epoch=1,
            seed=3,
            dedup_level="video",
            on_exhaustion="warn_and_relax",
        )
        batches = list(sampler)
        self.assertEqual(len(batches[0]), 4)
        self.assertGreater(sampler.last_epoch_dedup_failures, 0)

    def test_legacy_boolean_maps_to_level(self) -> None:
        entries = _videos(count=2, starts=1)
        disabled = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=4,
            steps_per_epoch=1,
            seed=3,
            deduplicate_within_global_batch=False,
        )
        self.assertEqual(disabled.dedup_level, "none")
        enabled = VideoBalancedPairBatchSampler(
            entries,
            local_batch_size=4,
            steps_per_epoch=1,
            seed=3,
            deduplicate_within_global_batch=True,
        )
        self.assertEqual(enabled.dedup_level, "pair")


if __name__ == "__main__":
    unittest.main()
