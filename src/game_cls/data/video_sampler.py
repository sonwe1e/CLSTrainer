from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence

from .lazy_pair_dataset import PairRequest
from .video_index import VideoEntry


def _choice(rng: random.Random, values: Sequence, weights: Sequence[float]):
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


class VideoBalancedPairBatchSampler:
    """Delta-first, deterministic sampler without materializing PairSample objects."""

    def __init__(
        self,
        videos: Sequence[VideoEntry],
        local_batch_size: int,
        steps_per_epoch: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        game_alpha: float = 0.25,
        class_probability: dict[int, float] | None = None,
        delta_probability: dict[int, float] | None = None,
        deduplicate_within_global_batch: bool | None = None,
        dedup_level: str = "pair",
        on_exhaustion: str = "warn_and_relax",
    ) -> None:
        if local_batch_size <= 0 or steps_per_epoch <= 0:
            raise ValueError("batch size and steps_per_epoch must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if dedup_level not in ("none", "pair", "video"):
            raise ValueError(
                f"dedup_level must be none|pair|video; got {dedup_level!r}"
            )
        if on_exhaustion not in ("error", "warn_and_relax"):
            raise ValueError(
                f"on_exhaustion must be error|warn_and_relax; "
                f"got {on_exhaustion!r}"
            )
        if deduplicate_within_global_batch is not None:
            # Legacy boolean: True -> pair, False -> none. The explicit
            # dedup_level wins when both are given.
            if dedup_level == "pair" and not deduplicate_within_global_batch:
                dedup_level = "none"
            if dedup_level == "none" and deduplicate_within_global_batch:
                dedup_level = "pair"
        self.videos = videos
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
        self.start_step = 0
        self.last_epoch_dedup_failures = 0
        self.last_epoch_delta_counts: Counter[int] = Counter()
        self.last_epoch_game_label_delta_counts: Counter[
            tuple[str, int, int]
        ] = Counter()
        self._support: dict[int, dict[str, dict[int, list[int]]]] = {}
        for video_index, video in enumerate(videos):
            for delta, starts in video.valid_start_positions.items():
                if len(starts):
                    self._support.setdefault(delta, {}).setdefault(
                        video.game, {}
                    ).setdefault(video.label, []).append(video_index)
        if not self._support:
            raise ValueError("No legal training pairs are available")

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        self.epoch = epoch
        self.start_step = start_step

    def state_dict(self, step_in_epoch: int) -> dict:
        return {
            "epoch": self.epoch,
            "step_in_epoch": step_in_epoch,
            "seed": self.seed,
        }

    def __len__(self) -> int:
        return max(0, self.steps_per_epoch - self.start_step)

    def _sample_for_delta(self, rng: random.Random, delta: int) -> PairRequest:
        games = list(self._support[delta])
        game_weights = []
        for game in games:
            count = sum(
                len(video_indices)
                for video_indices in self._support[delta][game].values()
            )
            game_weights.append(max(1, count) ** self.game_alpha)
        game = _choice(rng, games, game_weights)
        labels = list(self._support[delta][game])
        label = _choice(
            rng, labels, [self.class_probability.get(item, 0.0) for item in labels]
        )
        video_index = rng.choice(self._support[delta][game][label])
        valid_starts = self.videos[video_index].valid_start_positions[delta]
        start_position = int(rng.choice(valid_starts))
        return PairRequest(
            video_index=video_index,
            delta=delta,
            start_position=start_position,
            augmentation_seed=rng.getrandbits(63),
        )

    def __iter__(self) -> Iterator[list[PairRequest]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        self.last_epoch_delta_counts = Counter()
        self.last_epoch_game_label_delta_counts = Counter()
        self.last_epoch_dedup_failures = 0
        global_batch_size = self.local_batch_size * self.world_size
        deltas = list(self._support)
        delta_weights = [self.delta_probability.get(item, 0.0) for item in deltas]
        for step in range(self.steps_per_epoch):
            selected: list[PairRequest] = []
            used: set[tuple[int, int, int] | int] = set()
            attempts = 0
            max_attempts = max(100, global_batch_size * 20)
            dedup_active = self.dedup_level != "none"
            while len(selected) < global_batch_size:
                delta = _choice(rng, deltas, delta_weights)
                request = self._sample_for_delta(rng, delta)
                if self.dedup_level == "video":
                    identity: tuple[int, int, int] | int = request.video_index
                else:
                    identity = (
                        request.video_index,
                        request.delta,
                        request.start_position,
                    )
                attempts += 1
                if dedup_active and identity in used:
                    if attempts < max_attempts:
                        # Re-sample INSIDE the same delta so the effective
                        # per-delta distribution stays the configured one.
                        while identity in used and attempts < max_attempts:
                            request = self._sample_for_delta(rng, delta)
                            identity = (
                                request.video_index
                                if self.dedup_level == "video"
                                else (
                                    request.video_index,
                                    request.delta,
                                    request.start_position,
                                )
                            )
                            attempts += 1
                            self.last_epoch_dedup_failures += 1
                    elif self.on_exhaustion == "error":
                        raise RuntimeError(
                            "Deduplication exhausted: could not fill a "
                            "global batch without repeating "
                            f"{self.dedup_level} identities after "
                            f"{max_attempts} attempts (batch size "
                            f"{global_batch_size}). Reduce the batch size "
                            "or set data.deduplication.on_exhaustion=warn_and_relax."
                        )
                    else:
                        self.last_epoch_dedup_failures += 1
                selected.append(request)
                used.add(identity)
                self.last_epoch_delta_counts[request.delta] += 1
                video = self.videos[request.video_index]
                self.last_epoch_game_label_delta_counts[
                    (video.game, video.label, request.delta)
                ] += 1
            if step < self.start_step:
                continue
            start = self.rank * self.local_batch_size
            yield selected[start : start + self.local_batch_size]


class DeterministicIndexBatchSampler:
    """Exact-resume sampler used by the synthetic smoke dataset."""

    def __init__(
        self,
        dataset_size: int,
        local_batch_size: int,
        steps_per_epoch: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        self.dataset_size = dataset_size
        self.local_batch_size = local_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.start_step = 0

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        self.epoch = epoch
        self.start_step = start_step

    def state_dict(self, step_in_epoch: int) -> dict:
        return {"epoch": self.epoch, "step_in_epoch": step_in_epoch, "seed": self.seed}

    def __len__(self) -> int:
        return max(0, self.steps_per_epoch - self.start_step)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        global_batch_size = self.local_batch_size * self.world_size
        for step in range(self.steps_per_epoch):
            batch = [rng.randrange(self.dataset_size) for _ in range(global_batch_size)]
            if step < self.start_step:
                continue
            start = self.rank * self.local_batch_size
            yield batch[start : start + self.local_batch_size]
