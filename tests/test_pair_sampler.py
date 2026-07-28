from __future__ import annotations

from collections import Counter
import unittest

from game_cls.data.pair_sampler import BalancedDistributedPairBatchSampler
from game_cls.data.records import PairSample


def make_pairs() -> list[PairSample]:
    pairs = []
    for game in ("small", "large"):
        video_count = 1 if game == "small" else 4
        for label in (0, 1):
            for video in range(video_count):
                for delta in (1, 2, 3):
                    for start in range(20):
                        pairs.append(
                            PairSample(
                                game=game,
                                label=label,
                                video_id=f"{video:02d}",
                                frame0_id=start,
                                frame1_id=start + delta,
                                delta=delta,
                                image0_path="a.png",
                                image1_path="b.png",
                            )
                        )
    return pairs


class PairSamplerTests(unittest.TestCase):
    def test_rank_slices_are_disjoint_and_deterministic(self) -> None:
        pairs = make_pairs()
        kwargs = dict(
            pairs=pairs,
            local_batch_size=8,
            steps_per_epoch=5,
            world_size=2,
            seed=7,
        )
        rank0 = list(BalancedDistributedPairBatchSampler(rank=0, **kwargs))
        rank1 = list(BalancedDistributedPairBatchSampler(rank=1, **kwargs))
        self.assertEqual(
            rank0, list(BalancedDistributedPairBatchSampler(rank=0, **kwargs))
        )
        for left, right in zip(rank0, rank1):
            self.assertTrue(set(left).isdisjoint(right))

    def test_delta_and_class_distribution(self) -> None:
        pairs = make_pairs()
        sampler = BalancedDistributedPairBatchSampler(
            pairs,
            local_batch_size=64,
            steps_per_epoch=300,
            seed=11,
            delta_probability={1: 0.15, 2: 0.70, 3: 0.15},
        )
        selected = [index for batch in sampler for index in batch]
        deltas = Counter(pairs[index].delta for index in selected)
        labels = Counter(pairs[index].label for index in selected)
        self.assertAlmostEqual(deltas[2] / len(selected), 0.70, delta=0.03)
        self.assertAlmostEqual(labels[1] / len(selected), 0.50, delta=0.03)


if __name__ == "__main__":
    unittest.main()

