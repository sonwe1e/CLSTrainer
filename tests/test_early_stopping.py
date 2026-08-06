"""Early stopping and multi-objective checkpoints (step2 plan P2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EarlyStoppingTests(unittest.TestCase):
    def test_plateau_stops_before_budget_and_restores_best(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/cuda_debug.yaml")
            config["experiment"]["output_dir"] = str(Path(directory) / "run")
            config["train"].update(
                {
                    "max_steps": 60,
                    "steps_per_epoch": 30,
                    "local_batch_size": 4,
                    "log_every_steps": 30,
                }
            )
            config["evaluation"].update(
                {
                    "train_probe_every_steps": 0,
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 20,
                    "val_full_at_end": False,
                }
            )
            config["early_stopping"] = {
                "enabled": True,
                "monitor": "selection_score",
                "mode": "max",
                "full_validation_only": True,
                "burn_in_steps": 0,
                "patience_evaluations": 1,
                "min_delta": 0.0,
                "restore_best": True,
            }
            # Deterministic evaluations: first improves, second does not.
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.4,
                    "global_f1_tau099": 0.4,
                    "cross_entropy": 1.2,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 40,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 1)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)
            early = result["evaluation_state"]["early_stopping"]
            self.assertEqual(early["stop_reason"], "validation_plateau")
            self.assertEqual(early["stopped_at_step"], 40)
            self.assertEqual(early["best_step"], 20)
            self.assertEqual(early["bad_evaluation_count"], 1)
            # Stopped well before the 60-step budget.
            self.assertEqual(result["global_step"], 40)
            checkpoints = Path(directory) / "run" / "checkpoints"
            for name in (
                "model_best_selection.pth",
                "model_best_val_loss.pth",
                "model_best_worst_game.pth",
                "model_last.pth",
            ):
                self.assertTrue((checkpoints / name).is_file(), f"missing {name}")
            status = Path(directory) / "run" / "status.json"
            payload = __import__("json").loads(status.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("early_stopped"))
            self.assertEqual(payload.get("early_stopped_step"), 40)

    def test_restore_best_loads_selection_weights(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/cuda_debug.yaml")
            config["experiment"]["output_dir"] = str(Path(directory) / "run")
            config["train"].update(
                {
                    "max_steps": 60,
                    "steps_per_epoch": 30,
                    "local_batch_size": 4,
                    "log_every_steps": 30,
                }
            )
            config["evaluation"].update(
                {
                    "train_probe_every_steps": 0,
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 20,
                    "val_full_at_end": False,
                }
            )
            config["early_stopping"] = {
                "enabled": True,
                "monitor": "selection_score",
                "mode": "max",
                "full_validation_only": True,
                "burn_in_steps": 0,
                "patience_evaluations": 1,
                "min_delta": 0.0,
                "restore_best": True,
            }
            fake_metrics = [
                {
                    "selection_score": 0.9,
                    "global_f1_tau099": 0.9,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.8,
                    "global_f1_tau099": 0.8,
                    "cross_entropy": 1.2,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 40,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 1)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)
            summary = __import__("json").loads(
                (Path(directory) / "run" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["restored_best"])
            self.assertEqual(
                summary["early_stopping"]["stop_reason"],
                "validation_plateau",
            )
            self.assertEqual(result["global_step"], 40)


if __name__ == "__main__":
    unittest.main()
