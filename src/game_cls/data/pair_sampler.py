from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from .pair_dataset import group_pair_indices
from .records import PairSample


def _weighted_choice(rng: random.Random, values: Sequence, weights: Sequence[float]):
    if not values:
        raise RuntimeError("Cannot sample from an empty population")
    if not any(weight > 0 for weight in weights):
        raise ValueError(
            "All sampling weights are <= 0; the distribution would be "
            "meaningless. Fix the weights in the config (e.g. "
            "class_probability / delta_probability) instead of relying "
            "on a silent uniform fallback."
        )
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
        dedup_level: str = "pair",
        on_exhaustion: str = "warn_and_relax",
    ) -> None:
        if local_batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch size and steps_per_epoch must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if dedup_level not in ("none", "pair"):
            raise ValueError(
                f"dedup_level must be none|pair; got {dedup_level!r} "
                "(video-level dedup requires the lazy video backend)"
            )
        if on_exhaustion not in ("error", "warn_and_relax"):
            raise ValueError(
                f"on_exhaustion must be error|warn_and_relax; got {on_exhaustion!r}"
            )
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
        self.dedup_level = dedup_level
        self.on_exhaustion = on_exhaustion
        self.epoch = 0
        self.last_epoch_dedup_failures = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_one(self, rng: random.Random) -> int:
        available_deltas = sorted(
            {
                delta
                for by_label in self.groups.values()
                for by_video in by_label.values()
                for by_delta in by_video.values()
                for delta in by_delta
            }
        )
        delta = _weighted_choice(
            rng,
            available_deltas,
            [self.delta_probability.get(item, 0.0) for item in available_deltas],
        )
        games = [
            game
            for game, by_label in self.groups.items()
            if any(
                delta in by_delta
                for by_video in by_label.values()
                for by_delta in by_video.values()
            )
        ]
        game_weights = []
        for game in games:
            video_count = len(
                {
                    (label, video)
                    for label, by_video in self.groups[game].items()
                    for video, by_delta in by_video.items()
                    if delta in by_delta
                }
            )
            game_weights.append(max(1, video_count) ** self.game_alpha)
        game = _weighted_choice(rng, games, game_weights)

        labels = [
            label
            for label, by_video in self.groups[game].items()
            if any(delta in by_delta for by_delta in by_video.values())
        ]
        label = _weighted_choice(
            rng, labels, [self.class_probability.get(label, 0.0) for label in labels]
        )
        videos = [
            video
            for video, by_delta in self.groups[game][label].items()
            if delta in by_delta
        ]
        video = rng.choice(videos)
        return rng.choice(self.groups[game][label][video][delta])

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        global_batch_size = self.local_batch_size * self.world_size
        self.last_epoch_dedup_failures = 0
        dedup_active = self.dedup_level != "none"
        for _ in range(self.steps_per_epoch):
            selected: list[int] = []
            used: set[int] = set()
            attempts = 0
            max_attempts = max(100, global_batch_size * 20)
            while len(selected) < global_batch_size:
                index = self._sample_one(rng)
                attempts += 1
                if dedup_active and index in used:
                    if attempts < max_attempts:
                        self.last_epoch_dedup_failures += 1
                        continue
                    if self.on_exhaustion == "error":
                        raise RuntimeError(
                            "Deduplication exhausted: could not fill a "
                            "global batch without repeating pair "
                            f"identities after {max_attempts} attempts."
                        )
                    self.last_epoch_dedup_failures += 1
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
        videos = {
            (label, video) for label, by_video in by_label.items() for video in by_video
        }
        weights[game] = len(videos) ** game_alpha
    denominator = sum(weights.values())
    return {game: weight / denominator for game, weight in weights.items()}
