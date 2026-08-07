from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ExactTrainingResumeTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_training(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = load_config("configs/recipes/example_debug.yaml")
            base["device"]["accelerator"] = "cpu"
            base["train"].update(
                {
                    "max_steps": 4,
                    "steps_per_epoch": 4,
                    "local_batch_size": 2,
                    "log_every_steps": 100,
                    "verify_frozen_parameters": True,
                }
            )
            base["evaluation"].update(
                {
                    "quick_test_every_steps": 0,
                    "full_test_every_steps": 0,
                    "full_test_at_end": False,
                }
            )
            base["checkpoint"]["save_last_every_steps"] = 2

            uninterrupted = copy.deepcopy(base)
            uninterrupted["experiment"]["output_dir"] = str(root / "full")
            run_training(uninterrupted)

            partial = copy.deepcopy(base)
            partial["train"]["stop_after_steps"] = 2
            partial["experiment"]["output_dir"] = str(root / "resumed")
            run_training(partial)

            resumed = copy.deepcopy(base)
            resumed["experiment"]["output_dir"] = str(root / "resumed")
            resumed["train"]["resume_path"] = str(
                root / "resumed" / "checkpoints" / "checkpoint_last.pth"
            )
            run_training(resumed)

            full_state = torch.load(
                root / "full" / "checkpoints" / "model_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                root / "resumed" / "checkpoints" / "model_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(full_state.keys(), resumed_state.keys())
            for key in full_state:
                self.assertTrue(torch.equal(full_state[key], resumed_state[key]), key)
            metric_steps = [
                json.loads(line)["step"]
                for line in (root / "resumed" / "train_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(metric_steps, [2, 4])


if __name__ == "__main__":
    unittest.main()
