from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...contracts.data import SamplingFeedback


class NoOpFeedbackStore:
    """Default feedback store that records nothing (USERPLAN §11.6).

    A future hard-mining policy can read training losses, evaluation errors, or
    human weights from a real store without changing the ``SamplingPolicy``
    interface.
    """

    def __init__(self) -> None:
        self._by_step: dict[int, list[SamplingFeedback]] = defaultdict(list)

    def add(self, feedback: SamplingFeedback) -> None:
        self._by_step[feedback.checkpoint_step].append(feedback)

    def for_step(self, step: int) -> list[SamplingFeedback]:
        return self._by_step.get(step, [])

    def clear(self) -> None:
        self._by_step.clear()


NoOpFeedbackStore.__module__ = "game_cls.data.sampling.feedback"
