from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluationScheduleTests(unittest.TestCase):
    def test_npu_interval_metrics_synchronize_device(self) -> None:
        from game_cls.engine.trainer import (
            _synchronize_device_for_metrics,
        )

        device = type("Device", (), {"type": "npu"})()
        with mock.patch.object(torch, "npu", create=True) as npu:
            _synchronize_device_for_metrics(device)
            npu.synchronize.assert_called_once_with()

    def test_full_step_skips_quick_and_final_result_can_be_best(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/cuda_debug.yaml")
            config["experiment"]["output_dir"] = str(Path(directory) / "run")
            config["train"].update(
                {
                    "max_steps": 2,
                    "steps_per_epoch": 2,
                    "local_batch_size": 2,
                    "log_every_steps": 100,
                }
            )
            config["evaluation"].update(
                {
                    "val_quick_every_steps": 1,
                    "val_quick_pairs_per_video": 2,
                    "val_full_every_steps": 2,
                    "val_full_at_end": True,
                }
            )
            config["checkpoint"]["save_last_every_steps"] = 2
            result = run_training(config)
            self.assertEqual(result["evaluation_state"]["quick_test_count"], 1)
            self.assertEqual(result["evaluation_state"]["full_test_count"], 1)
            reports = Path(directory) / "run" / "reports"
            quick_report = reports / "val_quick_step_00000001"
            full_report = reports / "val_full_step_00000002"
            self.assertTrue((quick_report / "metrics.json").is_file())
            self.assertTrue((quick_report / "false_positive.parquet").is_file())
            self.assertFalse((quick_report / "errors.html").exists())
            self.assertFalse((quick_report / "metrics_by_video.csv").exists())
            self.assertTrue((full_report / "errors.html").is_file())
            self.assertTrue((full_report / "metrics_by_video.csv").is_file())
            checkpoints = Path(directory) / "run" / "checkpoints"
            self.assertTrue(
                (
                    checkpoints
                    / "checkpoint_best_observed_dev_test_selection.pth"
                ).is_file()
            )
            self.assertTrue(
                (
                    checkpoints
                    / "model_best_observed_dev_test_selection.pth"
                ).is_file()
            )
            self.assertTrue(
                (
                    checkpoints
                    / "model_best_observed_dev_test_selection.metadata.json"
                ).is_file()
            )
            # Multi-objective checkpoints required by the train/val/test
            # protocol (step2 plan P2).
            self.assertTrue(
                (checkpoints / "model_best_selection.pth").is_file()
            )
            self.assertTrue(
                (checkpoints / "model_best_val_loss.pth").is_file()
            )
            self.assertTrue(
                (checkpoints / "model_best_worst_game.pth").is_file()
            )
            # Unified evaluation history (step2 plan P1).
            history_path = Path(directory) / "run" / "metrics" / "evaluation.jsonl"
            self.assertTrue(history_path.is_file())
            history = [
                json.loads(line)
                for line in history_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(history[0]["split"], "validation")
            self.assertEqual(history[0]["scope"], "quick")
            self.assertIn("selection_score", history[0])
            self.assertIn("objective_loss", history[0])
            self.assertIn("positive_margin_pass_rate", history[0])
            metric_rows = [
                json.loads(line)
                for line in (
                    Path(directory) / "run" / "train_metrics.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(metric_rows), 1)
            interval = metric_rows[0]
            self.assertEqual(interval["step"], 2)
            self.assertGreater(
                interval["interval_samples_per_second"], 0
            )
            self.assertGreater(interval["interval_step_time"], 0)
            self.assertGreaterEqual(interval["data_wait_ratio"], 0)
            self.assertLessEqual(interval["data_wait_ratio"], 1)
            self.assertGreater(interval["evaluation_seconds"], 0)
            self.assertGreater(interval["checkpoint_seconds"], 0)
            self.assertIn("threshold_loss", interval)
            self.assertIn("learning_rate", interval)
            self.assertIn("grad_norm", interval)


if __name__ == "__main__":
    unittest.main()
