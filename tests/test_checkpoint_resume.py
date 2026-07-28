from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the current interpreter")
class CheckpointTests(unittest.TestCase):
    def test_model_only_and_full_state_restore(self) -> None:
        from game_cls.engine.checkpoint import (
            restore_training_checkpoint,
            save_checkpoint_pair,
        )

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint_pair(
                directory,
                "last",
                model,
                optimizer,
                scheduler,
                None,
                epoch=2,
                global_step=17,
                best_metrics={"f1": 0.5},
                config={"test": True},
            )
            model_path = Path(directory) / "model_last_full.pth"
            self.assertTrue(model_path.is_file())
            self.assertTrue((Path(directory) / "checkpoint_last.pth").is_file())
            model_payload = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(model_payload["global_step"], 17)
            self.assertEqual(
                model_payload["artifact_role"], "full_model_snapshot"
            )
            with torch.no_grad():
                model.weight.zero_()
            state = restore_training_checkpoint(
                Path(directory) / "checkpoint_last.pth",
                model,
                optimizer,
                scheduler,
            )
            self.assertEqual(state["global_step"], 17)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))


if __name__ == "__main__":
    unittest.main()
