from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...contracts.data import SamplingCatalog, SamplingContext, SamplingPolicy


@dataclass(frozen=True)
class SampleRequest:
    """Compact sampler request consumed by :class:`PolicyBatchSampler`.

    Mirrors the existing :class:`PairRequest` so the legacy
    :class:`VideoBalancedPairBatchSampler` can be reused unchanged.
    """

    video_index: int
    delta: int
    start_position: int
    augmentation_seed: int = 0


class SamplingPolicyBase:
    """Convenience base exposing the protocol's required attributes."""

    policy_name = "base"
    state_version = 1

    def sample_global_batch(
        self, catalog: SamplingCatalog, context: SamplingContext
    ) -> list[Any]:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        del state

    def update_feedback(self, feedback: Any) -> None:
        del feedback
