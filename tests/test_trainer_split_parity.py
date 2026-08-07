"""P1 golden guard: a synthetic training run writes the canonical artifacts.

The old ``engine/trainer.py`` monolith was split into ``engine/training``.
This test pins the observable output contract of ``run_training`` — the
metric-name set, evaluation-history record keys, and run-dir artifacts — so
any future refactor of the training loop that changes what it emits (or
stops emitting) is caught here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the current interpreter")
class TrainerSplitParityTests(unittest.TestCase):
    def _run_short_training(self) -> Path:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        directory = tempfile.mkdtemp()
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
                "val_full_every_steps": 10,
                "val_full_at_end": False,
                "tensorboard_live": False,
            }
        )
        config["checkpoint"].update({"save_topk": 0})
        config.pop("early_stopping", None)
        run_training(config)
        return Path(directory) / "run"

    def test_train_metrics_record_has_canonical_keys(self) -> None:
        output_dir = self._run_short_training()
        records = [
            json.loads(line)
            for line in (output_dir / "train_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertTrue(records, "train_metrics.jsonl must not be empty")
        required = {
            "step",
            "interval_loss",
            "interval_ce",
            "interval_threshold_loss",
            "interval_threshold_weight",
            "interval_accuracy",
            "interval_samples",
            "interval_samples_per_second",
            "learning_rate",
            "grad_norm",
            "data_wait_ratio",
            "host_enqueue_timing",
        }
        missing = required - set(records[0])
        self.assertEqual(missing, set(), f"missing canonical metric keys: {missing}")

    def test_evaluation_history_and_run_artifacts(self) -> None:
        output_dir = self._run_short_training()
        # Full-validation record was written with the neutral metric names.
        history = output_dir / "metrics" / "evaluation.jsonl"
        self.assertTrue(history.is_file())
        record = json.loads(history.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["scope"], "full")
        self.assertIn("global_f1_at_decision_threshold", record)
        self.assertIn("selection_score", record)
        # Immutable run artifacts all present.
        for name in (
            "manifest.json",
            "status.json",
            "resolved_config.json",
            "summary.md",
            "overview.html",
            "training_summary.json",
        ):
            self.assertTrue(
                (output_dir / name).is_file(), f"missing run artifact {name}"
            )


if __name__ == "__main__":
    unittest.main()
