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

# A staged rule set: the second stage opens at step 4, so any checkpoint saved
# after that carries MORE optimizer parameter groups than a step-0 optimizer.
STAGED_RULES = {
    "cls_head": {"pattern": r"^cls\.", "lr_scale": 1.0, "unfreeze_at_step": 0},
    "backbone_late": {
        "pattern": r"^backbone\.0\.",
        "lr_scale": 0.1,
        "unfreeze_at_step": 4,
    },
}


@unittest.skipIf(torch is None, "torch is not installed")
class TrainableRulesResumeTests(unittest.TestCase):
    def _config(self, directory: str, resume_path: str | None = None) -> dict:
        from game_cls.config import load_config

        config = load_config("configs/recipes/example_debug.yaml")
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

    def test_resume_past_unfreeze_boundary_rebuilds_matching_groups(self) -> None:
        """A checkpoint saved past a boundary has more optimizer groups than a
        step-0 optimizer. Resume must rebuild the optimizer for the SAVED step
        before loading, or ``load_state_dict`` raises "loaded state dict has a
        different number of parameter groups"."""
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            config["model"]["trainable_rules"] = STAGED_RULES
            result = run_training(config)
            run_dir = Path(result["output_dir"])
            checkpoint_path = run_dir / "checkpoints" / "checkpoint_last.pth"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Saved past step 4, so the late stage is live in both the mask and
            # the optimizer state.
            self.assertIn("backbone.0.weight", saved["trainable_state"])
            groups_at_save = len(saved["optimizer"]["param_groups"])
            self.assertGreater(groups_at_save, 2)

            resume_config = self._config(directory)
            resume_config["model"]["trainable_rules"] = STAGED_RULES
            resume_config["train"]["resume_path"] = str(checkpoint_path)
            resume_config["train"]["max_steps"] = 18
            result2 = run_training(resume_config)
            self.assertGreaterEqual(int(result2["global_step"]), 18)
            resumed = torch.load(
                run_dir / "checkpoints" / "checkpoint_last.pth",
                map_location="cpu",
                weights_only=False,
            )
            # The group layout survived the round trip -- no silent regrouping.
            self.assertEqual(len(resumed["optimizer"]["param_groups"]), groups_at_save)
            self.assertIn("backbone.0.weight", resumed["trainable_state"])

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
