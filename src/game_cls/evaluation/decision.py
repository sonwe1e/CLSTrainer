from __future__ import annotations

from typing import Any

from ..contracts.evaluation import DecisionOutput, DecisionPolicy
from ..contracts.task import PredictionBatch
from ..losses.threshold_loss import probability_threshold_to_margin


class BinaryThresholdDecision(DecisionPolicy):
    """Default decision: ``sigmoid(margin) > threshold`` with a strict ``>``.

    ``margin = logit1 - logit0``. The strict-greater-than behavior is part of
    the business contract and must not change (USERPLAN §10.2).
    """

    policy_name = "threshold"

    def __init__(self, threshold: float = 0.99) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0,1), got {threshold}")
        self.threshold = threshold
        self._cutoff = probability_threshold_to_margin(threshold)

    def decide(self, prediction_batch: PredictionBatch) -> DecisionOutput:
        import torch

        margins = prediction_batch.extras["margins"]
        probabilities = torch.sigmoid(margins)
        predictions = margins > self._cutoff
        return DecisionOutput(
            scores=probabilities,
            predictions=predictions,
            auxiliary={
                "margins": margins,
                "threshold": self.threshold,
                "cutoff": self._cutoff,
            },
        )
