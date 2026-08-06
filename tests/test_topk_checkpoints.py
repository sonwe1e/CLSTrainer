"""Top-K checkpoint registry (step3 P0-1b)."""

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
class TopKCheckpointTests(unittest.TestCase):
    def _config(self, directory: str, save_topk: int) -> dict:
        from game_cls.config import load_config

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
        config["checkpoint"].update(
            {
                "save_topk": save_topk,
                "topk_monitor": "selection_score",
            }
        )
        config.pop("early_stopping", None)
        return config

    def test_keeps_two_best_and_evicts_worst(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, save_topk=2)
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.9,
                    "global_f1_tau099": 0.9,
                    "cross_entropy": 0.8,
                    "worst_game_f1_tau099": 0.7,
                    "checkpoint_step": 40,
                },
                {
                    "selection_score": 0.4,
                    "global_f1_tau099": 0.4,
                    "cross_entropy": 1.2,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 60,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 2)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)

            registry = result["evaluation_state"]["topk_registry"]
            # Sorted best-first: step 40 (0.9), step 20 (0.5); step 60 (0.4)
            # must have been evicted.
            self.assertEqual([entry["step"] for entry in registry], [40, 20])
            self.assertEqual(registry[0]["value"], 0.9)
            self.assertEqual(registry[1]["value"], 0.5)

            checkpoints = Path(directory) / "run" / "checkpoints"
            for step in (20, 40):
                self.assertTrue(
                    (checkpoints / f"model_topk_{step:08d}.pth").is_file(),
                    f"missing topk file for step {step}",
                )
                self.assertTrue(
                    (checkpoints / f"checkpoint_topk_{step:08d}.pth").is_file()
                )
            # The evicted checkpoint must be gone.
            self.assertFalse((checkpoints / "model_topk_00000060.pth").exists())
            # Registry persisted next to the checkpoints.
            registry_file = checkpoints / "topk_registry.json"
            self.assertTrue(registry_file.is_file())
            persisted = json.loads(registry_file.read_text(encoding="utf-8"))
            self.assertEqual(persisted["monitor"], "selection_score")
            self.assertEqual(
                [entry["step"] for entry in persisted["entries"]], [40, 20]
            )
            # Summary payload carries the list for run show / summary.md.
            summary = json.loads(
                (Path(directory) / "run" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [entry["step"] for entry in summary["topk_checkpoints"]],
                [40, 20],
            )

    def test_lower_better_monitor_keeps_lowest_cross_entropy(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, save_topk=2)
            config["checkpoint"]["topk_monitor"] = "cross_entropy"
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.9,
                    "global_f1_tau099": 0.9,
                    "cross_entropy": 0.6,
                    "worst_game_f1_tau099": 0.7,
                    "checkpoint_step": 40,
                },
                {
                    "selection_score": 0.4,
                    "global_f1_tau099": 0.4,
                    "cross_entropy": 0.8,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 60,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 2)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)

            registry = result["evaluation_state"]["topk_registry"]
            # Lowest CE first: step 40 (0.6), then step 60 (0.8); step 20 (1.0)
            # evicted.
            self.assertEqual([entry["step"] for entry in registry], [40, 60])
            checkpoints = Path(directory) / "run" / "checkpoints"
            self.assertTrue((checkpoints / "model_topk_00000040.pth").is_file())
            self.assertTrue((checkpoints / "model_topk_00000060.pth").is_file())
            self.assertFalse((checkpoints / "model_topk_00000020.pth").exists())

    def test_disabled_topk_writes_no_topk_files(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, save_topk=0)
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                }
            ]

            def fake_run_evaluation(**kwargs):
                return mock.Mock(metrics=dict(fake_metrics[0]))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)
            self.assertEqual(result["evaluation_state"]["topk_registry"], [])
            checkpoints = Path(directory) / "run" / "checkpoints"
            self.assertFalse(list(checkpoints.glob("model_topk_*.pth")))


if __name__ == "__main__":
    unittest.main()
