"""Unified evaluation history + train probe (step2 plan P1)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluationHistoryTests(unittest.TestCase):
    def test_train_probe_and_gap_records_are_written(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.training.loop import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/recipes/example_debug.yaml")
            config["experiment"]["output_dir"] = str(Path(directory) / "run")
            config["train"].update(
                {
                    "max_steps": 100,
                    "steps_per_epoch": 50,
                    "local_batch_size": 4,
                    "log_every_steps": 25,
                }
            )
            config["evaluation"].update(
                {
                    "train_probe_every_steps": 25,
                    "train_probe_pairs_per_video": 2,
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 50,
                    "val_full_at_end": False,
                }
            )
            result = run_training(config)
            history_path = Path(result["output_dir"]) / "metrics" / "evaluation.jsonl"
            self.assertTrue(history_path.is_file())
            history = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["contract_version"] == 5 for row in history))
            kinds = [row["kind"] for row in history]
            self.assertIn("train_probe", kinds)
            self.assertIn("val_full", kinds)
            probe_rows = [row for row in history if row["kind"] == "train_probe"]
            full_rows = [row for row in history if row["kind"] == "val_full"]
            self.assertTrue(probe_rows)
            self.assertTrue(full_rows)
            self.assertTrue(all(row["split"] == "train_probe" for row in probe_rows))
            self.assertTrue(
                all(
                    row["split"] == "validation" and row["scope"] == "full"
                    for row in full_rows
                )
            )
            # Step 50 runs both probe and full: the gap must be recorded.
            gap_rows = [row for row in full_rows if "generalization_ce_gap" in row]
            self.assertTrue(gap_rows, "no generalization_ce_gap recorded")
            self.assertIn("generalization_score_gap", gap_rows[0])
            # Margin statistics from the evaluator.
            self.assertIn("positive_margin_pass_rate", gap_rows[0])
            self.assertIn("negative_margin_p50", gap_rows[0])
            # Selection score is annotated on every row.
            self.assertTrue(
                all(
                    isinstance(row.get("selection_score"), (int, float))
                    for row in history
                )
            )

    def test_interval_metrics_are_sample_weighted(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.training.loop import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/recipes/example_debug.yaml")
            config["experiment"]["output_dir"] = str(Path(directory) / "run")
            config["train"].update(
                {
                    "max_steps": 20,
                    "steps_per_epoch": 20,
                    "local_batch_size": 4,
                    "log_every_steps": 10,
                }
            )
            config["evaluation"].update(
                {
                    "train_probe_every_steps": 0,
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 0,
                    "val_full_at_end": False,
                }
            )
            result = run_training(config)
            run_dir = Path(result["output_dir"])
            rows = [
                json.loads(line)
                for line in (run_dir / "train_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["contract_version"], 5)
                for key in (
                    "interval_loss",
                    "interval_ce",
                    "interval_threshold_loss",
                    "interval_threshold_weight",
                    "interval_accuracy",
                    "interval_positive_recall_at_decision_threshold",
                    "interval_negative_specificity_at_decision_threshold",
                    "interval_samples",
                ):
                    self.assertIn(key, row)
                # Raw last-batch values are kept alongside interval means.
                for key in ("loss", "ce", "threshold_loss"):
                    self.assertIn(key, row)
                self.assertEqual(row["interval_samples"], 40)


if __name__ == "__main__":
    unittest.main()
