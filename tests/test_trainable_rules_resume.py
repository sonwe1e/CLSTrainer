"""trainable_rules checkpoint identity and resume (step5 P4)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


RULES = {"cls_head": {"pattern": r"^cls\.", "lr_scale": 1.0, "unfreeze_at_step": 0}}


@unittest.skipIf(torch is None, "torch is not installed")
class TrainableRulesResumeTests(unittest.TestCase):
    def _config(self, directory: str, resume_path: str | None = None) -> dict:
        from game_cls.config import load_config

        config = load_config("configs/cuda_debug.yaml")
        config["experiment"]["output_dir"] = str(Path(directory) / "run")
        config["model"]["trainable_rules"] = RULES
        config["train"].update(
            {
                "max_steps": 12,
                "steps_per_epoch": 12,
                "local_batch_size": 4,
                "log_every_steps": 6,
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
        config["checkpoint"].update(
            {"save_last_every_steps": 6, "save_topk": 0, "save_best_selection": False}
        )
        config.pop("early_stopping", None)
        if resume_path:
            config["train"]["resume_path"] = resume_path
        return config

    def test_rules_run_saves_fingerprint_and_resumes(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            result = run_training(config)
            run_dir = Path(result["output_dir"])
            # Checkpoint carries the rule fingerprint + trainable state.
            checkpoint = torch.load(
                run_dir / "checkpoints" / "checkpoint_last.pth",
                map_location="cpu",
                weights_only=False,
            )
            self.assertTrue(checkpoint.get("trainable_rules_fingerprint"))
            self.assertIn("cls.weight", checkpoint.get("trainable_state"))

            # Resume with the same rules and a longer budget (extend).
            resume_config = self._config(directory)
            resume_config["train"]["resume_path"] = str(
                run_dir / "checkpoints" / "checkpoint_last.pth"
            )
            resume_config["train"]["max_steps"] = 18
            result2 = run_training(resume_config)
            self.assertGreaterEqual(int(result2["global_step"]), 18)

    def test_changed_rules_block_resume(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            result = run_training(config)
            run_dir = Path(result["output_dir"])
            resume_config = self._config(directory)
            resume_config["train"]["resume_path"] = str(
                run_dir / "checkpoints" / "checkpoint_last.pth"
            )
            # Editing the rule set must refuse the resume.
            resume_config["model"]["trainable_rules"] = {
                "cls_head": {
                    "pattern": r"^cls\.",
                    "lr_scale": 0.5,
                    "unfreeze_at_step": 0,
                }
            }
            with self.assertRaises(RuntimeError):
                run_training(resume_config)

    def test_fingerprint_recorded_in_manifest_safe(self) -> None:
        # Sanity: the rule dict is JSON-serializable (it ships in resolved config).
        from game_cls.config_schema import finalize_config

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            finalized = finalize_config(config)
            json.dumps(finalized)  # must not raise


if __name__ == "__main__":
    unittest.main()
