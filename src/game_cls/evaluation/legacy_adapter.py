from __future__ import annotations

from typing import Any

from ..contracts.evaluation import EvaluatorSuite
from ..engine.evaluator import EvaluationOutput, evaluate
from ..registry import register


@register("evaluation_suite", "legacy_binary")
def build_legacy_evaluator_suite(**kwargs: Any) -> LegacyEvaluatorSuite:
    """Factory registered as ``evaluation_suite/legacy_binary``."""
    return LegacyEvaluatorSuite(**kwargs)


class LegacyEvaluatorSuite(EvaluatorSuite):
    """Wraps the battle-tested legacy ``evaluate`` behind the new suite protocol.

    The new ``BinaryThresholdEvaluatorSuite`` has device and distributed bugs
    (CPU accumulators receiving GPU tensors, CPU-side all-reduces, quick eval
    dropping false negatives). Until those are resolved and golden-tested, the
    runner talks to the evaluator through this adapter, which delegates to the
    existing ``evaluate`` function that is known to work on CPU/CUDA/NPU and
    across process groups.
    """

    #: Task names this evaluator suite supports. The builder checks this and
    #: raises at startup if the configured task is incompatible (fail-fast).
    supported_task_names: set[str] = {"dual_frame_binary"}

    suite_name = "legacy_binary"

    def __init__(
        self,
        *,
        threshold: float = 0.99,
        amp: bool = False,
        amp_dtype: str = "bfloat16",
        full_auc_mode: str = "histogram",
        auc_histogram_bins: int = 4096,
        quick_error_limit: int = 200,
        parquet_row_group_size: int = 4096,
    ) -> None:
        self._threshold = threshold
        self._amp = amp
        self._amp_dtype = amp_dtype
        self._full_auc_mode = full_auc_mode
        self._auc_histogram_bins = auc_histogram_bins
        self._quick_error_limit = quick_error_limit
        self._parquet_row_group_size = parquet_row_group_size

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
    ) -> Any:
        device = runtime.accelerator.device
        dist = runtime.distributed
        result = evaluate(
            model,
            dataloader,
            device,
            self._threshold,
            checkpoint_step=checkpoint_step,
            distributed=dist.world_size > 1,
            rank=dist.rank,
            world_size=dist.world_size,
            evaluation_kind=evaluation_kind,
            report_dir=output_dir,
            full_auc_mode=self._full_auc_mode,
            auc_histogram_bins=self._auc_histogram_bins,
            quick_error_limit=self._quick_error_limit,
            amp=self._amp,
            amp_dtype=self._amp_dtype,
            parquet_row_group_size=self._parquet_row_group_size,
            group_catalogs=getattr(
                getattr(dataloader, "dataset", None), "group_catalogs", None
            ),
        )
        return _SuiteResult(
            metrics=result.metrics,
            grouped_metrics=result.grouped_metrics,
            near_threshold=result.near_threshold,
        )


class _SuiteResult:
    """Mirror of the old ``EvaluationOutput`` minus ``errors`` (the suite
    protocol surfaces ``near_threshold`` separately)."""

    def __init__(
        self,
        metrics: dict | None,
        grouped_metrics: dict | None,
        near_threshold: list[dict],
    ) -> None:
        self.metrics = metrics
        self.grouped_metrics = grouped_metrics
        self.near_threshold = near_threshold
