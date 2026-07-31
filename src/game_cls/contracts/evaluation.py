from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DecisionOutput:
    scores: Any
    predictions: Any
    auxiliary: dict[str, Any] = field(default_factory=dict)


class DecisionPolicy(Protocol):
    @property
    def policy_name(self) -> str: ...

    def decide(self, prediction_batch: Any) -> DecisionOutput: ...


class MetricAccumulator(Protocol):
    def update(self, prediction_batch: Any, decision: DecisionOutput) -> None: ...

    def distributed_reduce(self, runtime: Any) -> None: ...

    def compute(self) -> dict[str, Any]: ...


class GroupAggregator(Protocol):
    def update(self, prediction_batch: Any, decision: DecisionOutput) -> None: ...

    def distributed_reduce(self, runtime: Any) -> None: ...

    def compute(self) -> dict[str, Any]: ...


@dataclass
class ErrorBatch:
    false_positives: list[dict]
    false_negatives: list[dict]
    near_threshold: list[dict]


class ErrorExtractor(Protocol):
    def extract(
        self, prediction_batch: Any, decision: DecisionOutput, checkpoint_step: int
    ) -> ErrorBatch: ...


class ReportWriter(Protocol):
    def write(
        self,
        output_dir: Any,
        metrics: dict[str, Any],
        grouped_metrics: dict[str, Any],
        errors: ErrorBatch,
        metadata: dict[str, Any],
    ) -> None: ...


class EvaluatorSuite(Protocol):
    """Orchestrates model-output -> score -> decision -> metrics -> groups -> errors -> report."""

    def evaluate(
        self,
        model: Any,
        dataloader: Any,
        runtime: Any,
        task: Any,
        context: Any,
    ) -> Any: ...
