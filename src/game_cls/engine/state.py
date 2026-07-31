from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts.task import StepContext


@dataclass
class ExperimentState:
    """Mutable training state tracked by the :class:`ExperimentRunner`."""

    global_step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    best_metrics: dict[str, Any] = field(default_factory=dict)
    evaluation_state: dict[str, Any] = field(default_factory=dict)
    processed_samples: int = 0


def make_step_context(state: ExperimentState, epoch: int, device: Any, use_amp: bool, amp_dtype: str) -> StepContext:
    return StepContext(
        global_step=state.global_step,
        total_steps=state.total_steps,
        epoch=epoch,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
