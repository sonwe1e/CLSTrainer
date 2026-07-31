from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.evaluation import DecisionOutput, ErrorBatch, MetricAccumulator, ReportWriter


class BinaryReportWriter(ReportWriter):
    """Writes the standard evaluation report files (USERPLAN §10.5).

    Delegates to the existing :mod:`game_cls.reports.error_writer` for the
    actual file serialization, so the report schema, CSV columns and HTML
    preview stay byte-for-byte compatible.
    """

    def __init__(self, *, html_max_errors: int = 200, lightweight: bool = False) -> None:
        self._html_max_errors = html_max_errors
        self._lightweight = lightweight

    def write(
        self,
        output_dir: Any,
        metrics: dict[str, Any],
        grouped_metrics: dict[str, Any],
        errors: ErrorBatch,
        metadata: dict[str, Any],
    ) -> None:
        from game_cls.reports.error_writer import write_evaluation_report

        del metadata
        all_errors = errors.false_positives + errors.false_negatives
        write_evaluation_report(
            output_dir,
            metrics,
            grouped_metrics,
            errors=all_errors,
            near_threshold=errors.near_threshold,
            merge_shards=False,
            html_max_errors=self._html_max_errors,
            lightweight=self._lightweight,
            preview_decoder=None,
        )
