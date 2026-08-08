from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace

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
        dedup_level: str = "pair",
        on_exhaustion: str = "warn_and_relax",
        hard_negative_cfg: dict | None = None,
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
                f"on_exhaustion must be error|warn_and_relax; got {on_exhaustion!r}"
            )
        if hard_negative_cfg and hard_negative_cfg.get("max_pairs_per_video"):
            # Deterministic per-video cap: only the first N start positions
            # of each video are eligible, so the model cannot memorize a few
            # scenes by revisiting every position of one video.
            cap = int(hard_negative_cfg["max_pairs_per_video"])
            videos = [
                replace(
                    video,
                    valid_start_positions={
                        delta: starts[:cap]
                        for delta, starts in video.valid_start_positions.items()
                    },
                )
                for video in videos
            ]
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
        self.last_epoch_game_label_delta_counts: Counter[tuple[str, int, int]] = (
            Counter()
        )
        self._support: dict[int, dict[str, dict[int, list[int]]]] = {}
        for video_index, video in enumerate(videos):
            for delta, starts in video.valid_start_positions.items():
                if len(starts):
                    self._support.setdefault(delta, {}).setdefault(
                        video.game, {}
                    ).setdefault(video.label, []).append(video_index)
        if not self._support:
            raise ValueError("No legal training pairs are available")

        # Hard-negative subtype buckets (step5 P2). Only built when enabled;
        # otherwise sampling stays byte-identical. ``negative_subtype`` is
        # read from the sidecar-joined VideoEntry (defaults to None).
        self._hard_negative_enabled = bool(
            hard_negative_cfg and hard_negative_cfg.get("enabled", False)
        )
        self._subtype_buckets: dict[int, dict[str, dict[str, list[int]]]] = {}
        if self._hard_negative_enabled:
            assert hard_negative_cfg is not None
            subtype_field = str(
                hard_negative_cfg.get("subtype_field", "negative_subtype")
            )
            hard_subtypes = set(hard_negative_cfg.get("hard_subtypes") or [])
            ordinary_subtypes = set(hard_negative_cfg.get("ordinary_subtypes") or [])
            for delta, games in self._support.items():
                for game, labels in games.items():
                    if 0 not in labels:
                        continue
                    hard: list[int] = []
                    ordinary: list[int] = []
                    for video_index in labels[0]:
                        subtype = getattr(self.videos[video_index], subtype_field, None)
                        if subtype in hard_subtypes:
                            hard.append(video_index)
                        elif ordinary_subtypes and subtype not in ordinary_subtypes:
                            continue  # unclassified subtype: excluded from both
                        else:
                            ordinary.append(video_index)
                    self._subtype_buckets.setdefault(delta, {}).setdefault(game, {})[
                        "hard"
                    ] = hard
                    self._subtype_buckets[delta][game]["ordinary"] = ordinary
            self._negative_mix: dict[str, float] = dict(
                (hard_negative_cfg.get("negative_mix") or {})
                or {"ordinary": 0.5, "hard": 0.5}
            )
            self._min_videos_per_subtype_bucket = int(
                hard_negative_cfg.get("min_videos_per_subtype_bucket", 1)
            )

    def subtype_bucket_summary(self) -> dict:
        """Describe the hard-negative buckets this sampler actually built.

        Returns per-bucket distinct video counts *and* legal pair counts (the
        eligible start positions after ``max_pairs_per_video``), because an
        empty or tiny hard bucket is otherwise invisible: sampling falls back
        to the other bucket and training just looks normal. Also lists the
        under-sized (game, delta) cells that will take that fallback and the
        negatives excluded from both buckets by ``ordinary_subtypes``.
        """
        if not self._hard_negative_enabled:
            return {"enabled": False}
        # Video counts are distinct videos (a video appears in one bucket per
        # delta, so summing cells would multiply it by the delta count); pair
        # counts are sums, because a video contributes different start
        # positions at each delta.
        videos: dict[str, set[int]] = {"ordinary": set(), "hard": set()}
        pairs: dict[str, int] = {"ordinary": 0, "hard": 0}
        game_videos: dict[tuple[str, str], set[int]] = {}
        game_pairs: dict[tuple[str, str], int] = {}
        undersized: list[dict] = []
        excluded: set[int] = set()
        for delta, games in self._subtype_buckets.items():
            for game, buckets in games.items():
                classified: set[int] = set()
                for bucket in ("ordinary", "hard"):
                    members = buckets.get(bucket) or []
                    classified.update(members)
                    cell_pairs = sum(
                        len(self.videos[index].valid_start_positions.get(delta, ()))
                        for index in members
                    )
                    videos[bucket].update(members)
                    pairs[bucket] += cell_pairs
                    game_videos.setdefault((game, bucket), set()).update(members)
                    game_pairs[(game, bucket)] = (
                        game_pairs.get((game, bucket), 0) + cell_pairs
                    )
                    if len(members) < self._min_videos_per_subtype_bucket:
                        undersized.append(
                            {
                                "game": game,
                                "delta": int(delta),
                                "bucket": bucket,
                                "videos": len(members),
                            }
                        )
                excluded.update(
                    set(self._support[delta][game][0]) - classified,
                )
        by_game: dict[str, dict[str, dict[str, int]]] = {}
        for (game, bucket), gv_members in game_videos.items():
            by_game.setdefault(game, {})[bucket] = {
                "videos": len(gv_members),
                "legal_pairs": game_pairs.get((game, bucket), 0),
            }
        return {
            "enabled": True,
            "min_videos_per_subtype_bucket": self._min_videos_per_subtype_bucket,
            "negative_mix": dict(self._negative_mix),
            "buckets": {
                bucket: {
                    "videos": len(videos[bucket]),
                    "legal_pairs": pairs[bucket],
                }
                for bucket in ("ordinary", "hard")
            },
            "by_game": by_game,
            "undersized_cells": undersized,
            "excluded_videos": len(excluded),
        }

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
        if self._hard_negative_enabled and label == 0:
            video_index = self._sample_negative_video(rng, delta, game)
        else:
            candidates = self._support[delta][game][label]
            video_index = _choice(
                rng,
                candidates,
                [self.videos[index].sample_weight for index in candidates],
            )
        valid_starts = self.videos[video_index].valid_start_positions[delta]
        start_position = int(rng.choice(valid_starts))
        return PairRequest(
            video_index=video_index,
            delta=delta,
            start_position=start_position,
            augmentation_seed=rng.getrandbits(63),
        )

    def _sample_negative_video(self, rng: random.Random, delta: int, game: str) -> int:
        """Pick a negative video, mixing ordinary and hard subtype buckets.

        The bucket is chosen by ``negative_mix`` weight; an empty or
        under-sized bucket falls back to the other bucket so training never
        stalls. Falls back to the ungrouped candidate list when no subtype
        metadata exists for this (game, delta).
        """
        mix = self._negative_mix
        bucket = _choice(
            rng,
            ["ordinary", "hard"],
            [mix.get("ordinary", 0.0), mix.get("hard", 0.0)],
        )
        buckets = self._subtype_buckets.get(delta, {}).get(game, {})
        candidates = list(buckets.get(bucket) or [])
        if len(candidates) < self._min_videos_per_subtype_bucket:
            other = "hard" if bucket == "ordinary" else "ordinary"
            candidates = list(buckets.get(other) or [])
        if not candidates:
            candidates = list(self._support[delta][game][0])
        return int(
            _choice(
                rng,
                candidates,
                [self.videos[index].sample_weight for index in candidates],
            )
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
                    # The exhaustion policy is evaluated *after* the final
                    # retry. Previously a duplicate produced on that retry was
                    # appended silently because control never returned to the
                    # pre-retry ``elif``.
                    if identity in used and self.on_exhaustion == "error":
                        raise RuntimeError(
                            "Deduplication exhausted: could not fill a "
                            "global batch without repeating "
                            f"{self.dedup_level} identities after "
                            f"{max_attempts} attempts (batch size "
                            f"{global_batch_size}). Reduce the batch size "
                            "or set data.deduplication.on_exhaustion=warn_and_relax."
                        )
                    if identity in used:
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
