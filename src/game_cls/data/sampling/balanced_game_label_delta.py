from __future__ import annotations

from typing import Any

from ...contracts.data import SamplingCatalog, SamplingContext, SamplingFeedback, SamplingPolicy
from .base import SamplingPolicyBase


class BalancedGameLabelDeltaPolicy(SamplingPolicyBase):
    """Reproduces the current game/label/delta balanced sampling algorithm.

    This policy wraps the existing
    :class:`game_cls.data.video_sampler.VideoBalancedPairBatchSampler` so that the
    sampling algorithm lives behind the :class:`SamplingPolicy` interface
    (USERPLAN §11.4). The wrapped sampler is the single source of truth for the
    algorithm; this class only adapts its lifecycle to the policy protocol.
    """

    policy_name = "balanced_game_label_delta"
    state_version = 1

    def __init__(self, sampler: Any) -> None:
        self._sampler = sampler
        self._iterator = None
        self._iter_key: tuple[int, int] | None = None

    @classmethod
    def from_config(
        cls,
        videos: Any,
        local_batch_size: int,
        steps_per_epoch: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        game_alpha: float = 0.25,
        class_probability: dict[int, float] | None = None,
        delta_probability: dict[int, float] | None = None,
        deduplicate_within_global_batch: bool = True,
    ) -> "BalancedGameLabelDeltaPolicy":
        from game_cls.data.video_sampler import VideoBalancedPairBatchSampler

        sampler = VideoBalancedPairBatchSampler(
            videos,
            local_batch_size,
            steps_per_epoch,
            rank=rank,
            world_size=world_size,
            seed=seed,
            game_alpha=game_alpha,
            class_probability=class_probability,
            delta_probability=delta_probability,
            deduplicate_within_global_batch=deduplicate_within_global_batch,
        )
        return cls(sampler)

    def sample_rank_batch(
        self, catalog: SamplingCatalog, context: SamplingContext
    ) -> list[Any]:
        del catalog
        # The wrapped ``VideoBalancedPairBatchSampler`` already returns a
        # rank-local batch (it shards by rank/world_size internally). This
        # method therefore returns the *rank-local* batch, not a global batch.
        # Rebuild the iterator once per epoch, not once per step. Keying on
        # (epoch, step) would rebuild on every batch and turn epoch work O(N²).
        if context.step == 0 or self._iter_key != context.epoch or self._iterator is None:
            self._sampler.set_epoch(context.epoch, start_step=context.step)
            self._iterator = iter(self._sampler)
            self._iter_key = context.epoch
        try:
            return next(self._iterator)
        except StopIteration:
            # Exhausted the epoch's steps; restart from the current step.
            self._sampler.set_epoch(context.epoch, start_step=context.step)
            self._iterator = iter(self._sampler)
            self._iter_key = context.epoch
            return next(self._iterator)

    def state_dict(self) -> dict:
        return {"epoch": self._sampler.epoch, "start_step": self._sampler.start_step}

    def load_state_dict(self, state: dict) -> None:
        self._sampler.epoch = int(state.get("epoch", 0))
        self._sampler.start_step = int(state.get("start_step", 0))

    def update_feedback(self, feedback: list[SamplingFeedback]) -> None:
        del feedback

    @property
    def inner_sampler(self) -> Any:
        return self._sampler
