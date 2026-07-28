from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

from .pair_dataset import group_pair_indices
from .records import PairSample


def _weighted_choice(rng: random.Random, values: Sequence, weights: Sequence[float]):
    if not values:
        raise RuntimeError("Cannot sample from an empty population")
    return rng.choices(values, weights=weights, k=1)[0]


class BalancedDistributedPairBatchSampler:
    """Deterministic game -> class -> video -> delta -> pair batch sampler."""

    def __init__(
        self,
        pairs: Sequence[PairSample],
        local_batch_size: int,
        steps_per_epoch: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        game_alpha: float = 0.25,
        class_probability: dict[int, float] | None = None,
        delta_probability: dict[int, float] | None = None,
        deduplicate_within_global_batch: bool = True,
    ) -> None:
        if local_batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch size and steps_per_epoch must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.pairs = list(pairs)
        self.groups = group_pair_indices(pairs)
        if not self.groups:
            raise ValueError("No legal pairs are available")
        self.local_batch_size = local_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.game_alpha = game_alpha
        self.class_probability = class_probability or {0: 0.5, 1: 0.5}
        self.delta_probability = delta_probability or {1: 0.15, 2: 0.70, 3: 0.15}
        self.deduplicate = deduplicate_within_global_batch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_one(self, rng: random.Random) -> int:
        games = list(self.groups)
        game_weights = []
        for game in games:
            video_count = len(
                {
                    video
                    for by_video in self.groups[game].values()
                    for video in by_video
                }
            )
            game_weights.append(max(1, video_count) ** self.game_alpha)
        game = _weighted_choice(rng, games, game_weights)

        labels = list(self.groups[game])
        label = _weighted_choice(
            rng, labels, [self.class_probability.get(label, 0.0) for label in labels]
        )
        videos = list(self.groups[game][label])
        video = rng.choice(videos)
        available = self.groups[game][label][video]
        deltas = list(available)
        delta = _weighted_choice(
            rng, deltas, [self.delta_probability.get(delta, 0.0) for delta in deltas]
        )
        return rng.choice(available[delta])

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        global_batch_size = self.local_batch_size * self.world_size
        for _ in range(self.steps_per_epoch):
            selected: list[int] = []
            used: set[int] = set()
            attempts = 0
            max_attempts = max(100, global_batch_size * 20)
            while len(selected) < global_batch_size:
                index = self._sample_one(rng)
                attempts += 1
                if self.deduplicate and index in used and attempts < max_attempts:
                    continue
                selected.append(index)
                used.add(index)
            start = self.rank * self.local_batch_size
            yield selected[start : start + self.local_batch_size]


def expected_game_probabilities(
    pairs: Sequence[PairSample], game_alpha: float
) -> dict[str, float]:
    grouped = group_pair_indices(pairs)
    weights = {}
    for game, by_label in grouped.items():
        videos = {video for by_video in by_label.values() for video in by_video}
        weights[game] = len(videos) ** game_alpha
    denominator = sum(weights.values())
    return {game: weight / denominator for game, weight in weights.items()}

