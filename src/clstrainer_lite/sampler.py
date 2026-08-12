from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from typing import Protocol, Sequence

from torch.utils.data import Sampler


class _VideoIndexedDataset(Protocol):
    videos: Sequence

    def __len__(self) -> int: ...

    def position_count_for_video(self, index: int) -> int: ...


def _weighted_pick(rng: random.Random, items: Sequence, cumulative: Sequence[float]):
    value = rng.random() * cumulative[-1]
    return items[bisect.bisect_left(cumulative, value)]


def _cumulative(weights: Sequence[float]) -> tuple[float, ...]:
    total = 0.0
    result: list[float] = []
    for weight in weights:
        total += float(weight)
        result.append(total)
    if total <= 0:
        raise ValueError("sampling weights must sum to > 0")
    return tuple(result)


class BalancedVideoSampler(Sampler[int]):
    """Hierarchical game -> class -> video -> position sampling.

    The sampler is storage-agnostic: both ImagePairDataset and VideoPairDataset
    expose the same compact ``videos`` plus ``position_count_for_video`` view.
    Delta remains an online augmentation inside the dataset.

    ``game_balance_alpha`` interpolates between natural position frequency
    (0.0) and uniform games (1.0). Videos are sampled uniformly inside each
    game/class bucket so long videos cannot dominate merely by having more
    frames.
    """

    def __init__(
        self,
        dataset: _VideoIndexedDataset,
        *,
        class_probability: Sequence[float] = (0.5, 0.5),
        game_balance_alpha: float = 0.5,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if not 0.0 <= float(game_balance_alpha) <= 1.0:
            raise ValueError("game_balance_alpha must be in [0, 1]")
        if len(class_probability) != 2 or any(float(x) < 0 for x in class_probability):
            raise ValueError("class_probability must contain two non-negative values")
        if sum(float(x) for x in class_probability) <= 0:
            raise ValueError("class_probability must sum to > 0")

        self.dataset = dataset
        self.class_probability = tuple(float(x) for x in class_probability)
        self.game_balance_alpha = float(game_balance_alpha)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        offsets: list[int] = []
        total = 0
        buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
        game_mass: dict[str, int] = defaultdict(int)
        for video_index, video in enumerate(dataset.videos):
            count = int(dataset.position_count_for_video(video_index))
            if count <= 0:
                raise ValueError("dataset contains a video with no sampling positions")
            offsets.append(total)
            total += count
            buckets[(str(video.game), int(video.label))].append(video_index)
            game_mass[str(video.game)] += count
        if total != len(dataset):
            raise ValueError("dataset indexes must be contiguous video position ranges")
        self._offsets = tuple(offsets)
        self._counts = tuple(
            int(dataset.position_count_for_video(index)) for index in range(len(dataset.videos))
        )
        self._buckets = dict(buckets)

        self._games = tuple(sorted(game_mass))
        exponent = 1.0 - self.game_balance_alpha
        game_weights = [float(game_mass[game]) ** exponent for game in self._games]
        self._game_cumulative = _cumulative(game_weights)

        class_choices: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
        for game in self._games:
            labels = tuple(label for label in (0, 1) if (game, label) in self._buckets)
            weights = [self.class_probability[label] for label in labels]
            # A game with only one observed class remains sampleable; available
            # classes are renormalized rather than silently dropping the game.
            if sum(weights) <= 0:
                weights = [1.0] * len(labels)
            class_choices[game] = (labels, _cumulative(weights))
        self._class_choices = class_choices

        global_samples = int(samples_per_epoch) if samples_per_epoch is not None else len(dataset)
        if global_samples <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.global_samples_per_epoch = global_samples
        self.num_samples = math.ceil(global_samples / self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003 + self.rank * 97_409)
        for _ in range(self.num_samples):
            game = _weighted_pick(rng, self._games, self._game_cumulative)
            labels, class_cumulative = self._class_choices[game]
            label = _weighted_pick(rng, labels, class_cumulative)
            video_indices = self._buckets[(game, label)]
            video_index = video_indices[rng.randrange(len(video_indices))]
            local_position = rng.randrange(self._counts[video_index])
            yield self._offsets[video_index] + local_position

    def summary(self) -> dict:
        exponent = 1.0 - self.game_balance_alpha
        raw_weights = [
            self._game_cumulative[index]
            - (0.0 if index == 0 else self._game_cumulative[index - 1])
            for index in range(len(self._games))
        ]
        total = sum(raw_weights)
        return {
            "enabled": True,
            "strategy": "game->class->video->position",
            "game_balance_alpha": self.game_balance_alpha,
            "game_probability": {
                game: weight / total for game, weight in zip(self._games, raw_weights, strict=True)
            },
            "class_probability": {"0": self.class_probability[0], "1": self.class_probability[1]},
            "video_sampling": "uniform",
            "global_samples_per_epoch": self.global_samples_per_epoch,
            "samples_per_rank": self.num_samples,
            "world_size": self.world_size,
            "game_weight_exponent": exponent,
        }
