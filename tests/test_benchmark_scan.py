"""Benchmark gate checks and report writing (step5 P3/P6)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game_cls.reports.benchmark import check_gates, write_benchmark_report


class BenchmarkGateTests(unittest.TestCase):
    def test_fpr_gate_uses_upper_bound(self) -> None:
        metrics = {"global_fpr_at_decision_threshold": 0.008}
        checks = check_gates(metrics, {"global_fpr_at_decision_threshold": 0.01})
        self.assertTrue(checks[0][1])
        metrics["global_fpr_at_decision_threshold"] = 0.02
        checks = check_gates(metrics, {"global_fpr_at_decision_threshold": 0.01})
        self.assertFalse(checks[0][1])

    def test_recall_gate_uses_lower_bound(self) -> None:
        metrics = {"global_positive_recall_at_decision_threshold": 0.85}
        checks = check_gates(metrics, {"global_positive_recall_at_decision_threshold": 0.8})
        self.assertTrue(checks[0][1])
        metrics["global_positive_recall_at_decision_threshold"] = 0.7
        checks = check_gates(metrics, {"global_positive_recall_at_decision_threshold": 0.8})
        self.assertFalse(checks[0][1])

    def test_absent_metric_fails_gate(self) -> None:
        checks = check_gates({}, {"missing_metric": 0.01})
        self.assertFalse(checks[0][1])

    def test_report_written_with_scores_and_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = write_benchmark_report(
                Path(directory),
                run_id="run-1",
                checkpoint_alias="best_selection",
                metrics={"sample_count": 100, "global_fpr_at_decision_threshold": 0.005},
                gates=[("global_fpr_at_decision_threshold", True, "value=0.005 bound=0.01")],
                grouped_metrics=None,
            )
            self.assertTrue(report.is_file())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run-1")
            self.assertTrue(payload["gates"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
