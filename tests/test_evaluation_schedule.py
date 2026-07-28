from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluationScheduleTests(unittest.TestCase):
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
                    "quick_test_every_steps": 1,
                    "quick_test_pairs_per_video": 2,
                    "full_test_every_steps": 2,
                    "full_test_at_end": True,
                }
            )
            config["checkpoint"]["save_last_every_steps"] = 2
            result = run_training(config)
            self.assertEqual(result["evaluation_state"]["quick_test_count"], 1)
            self.assertEqual(result["evaluation_state"]["full_test_count"], 1)
            reports = Path(directory) / "run" / "reports"
            quick_report = reports / "quick_step_00000001"
            full_report = reports / "full_step_00000002"
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


if __name__ == "__main__":
    unittest.main()
