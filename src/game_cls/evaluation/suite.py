from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.evaluation import (
    DecisionPolicy,
    ErrorBatch,
    ErrorExtractor,
    EvaluatorSuite,
    GroupAggregator,
    MetricAccumulator,
    ReportWriter,
)
from ..contracts.task import StepContext
from .decision import BinaryThresholdDecision
from .errors import BinaryErrorExtractor
from .groups import BinaryGroupAggregator
from .metrics import BinaryConfusionAccumulator
from .reports import BinaryReportWriter


@dataclass
class _EvaluationResult:
    metrics: dict[str, Any] | None
    grouped_metrics: dict[str, Any] | None
    errors: Any


class BinaryThresholdEvaluatorSuite:
    """Orchestrates the full evaluation pipeline (USERPLAN §10.6).

    Executes ``Task.forward → Task.build_predictions → DecisionPolicy.decide →
    MetricAccumulator.update → GroupAggregator.update → ErrorExtractor.extract →
    ReportWriter.write``. The default components reproduce the existing
    ``evaluate`` behavior exactly; a future task supplies a different suite.
    """

    def __init__(
        self,
        task: Any,
        decision: DecisionPolicy | None = None,
        metrics: MetricAccumulator | None = None,
        groups: GroupAggregator | None = None,
        errors: ErrorExtractor | None = None,
        report: ReportWriter | None = None,
        *,
        auc_histogram_bins: int = 4096,
        quick_error_limit: int = 200,
        html_max_errors: int = 200,
        threshold: float = 0.99,
    ) -> None:
        self._task = task
        self._decision = decision or BinaryThresholdDecision(threshold=threshold)
        self._metrics = metrics or BinaryConfusionAccumulator(auc_histogram_bins=auc_histogram_bins)
        self._groups = groups or BinaryGroupAggregator()
        self._errors = errors or BinaryErrorExtractor(quick_error_limit=quick_error_limit)
        self._report = report or BinaryReportWriter(html_max_errors=html_max_errors)

    def evaluate(
        self,
        model: Any,
        dataloader: Any,
        runtime: Any,
        task: Any,
        context: Any,
        *,
        checkpoint_step: int = 0,
        evaluation_kind: str = "quick",
        output_dir: Any = None,
    ) -> _EvaluationResult:
        import torch

        model.eval()
        local_errors: list[dict] = []
        local_near: list[dict] = []
        quick_limit = self._errors._quick_limit if evaluation_kind == "quick" else 0
        try:
            with torch.inference_mode():
                for batch in dataloader:
                    device_batch = task.move_batch_to_device(batch, context)
                    task_output = task.forward(model, device_batch, context)
                    prediction_batch = task.build_predictions(task_output, device_batch, context)
                    decision = self._decision.decide(prediction_batch)
                    self._metrics.update(prediction_batch, decision)
                    self._groups.update(prediction_batch, decision)
                    error_batch = self._errors.extract(prediction_batch, decision, checkpoint_step)
                    if evaluation_kind == "quick":
                        local_errors.extend(error_batch.false_positives)
                        local_near.extend(error_batch.near_threshold)
                        if quick_limit > 0:
                            local_errors = local_errors[:quick_limit]
                            local_near = local_near[:quick_limit]
                    else:
                        local_errors.extend(error_batch.false_positives + error_batch.false_negatives)
                        local_near.extend(error_batch.near_threshold)
        finally:
            pass

        self._metrics.distributed_reduce(runtime)
        self._groups.distributed_reduce(runtime)

        metrics = None
        grouped = None
        if runtime.distributed.rank == 0 or runtime.distributed.world_size <= 1:
            metrics = self._metrics.compute()
            grouped = self._groups.compute()
            if output_dir is not None:
                false_positives = [e for e in local_errors if e.get("error_type") == "FP"]
                false_negatives = [e for e in local_errors if e.get("error_type") == "FN"]
                self._report.write(
                    output_dir,
                    metrics,
                    grouped,
                    errors=ErrorBatch(
                        false_positives=false_positives,
                        false_negatives=false_negatives,
                        near_threshold=local_near,
                    ),
                    metadata={"evaluation_kind": evaluation_kind, "checkpoint_step": checkpoint_step},
                )
        return _EvaluationResult(metrics=metrics, grouped_metrics=grouped, errors=local_errors)
