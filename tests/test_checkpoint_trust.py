"""Checkpoint trust boundary (step3 高优-6)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class CheckpointTrustTests(unittest.TestCase):
    def test_base_checkpoint_load_refuses_pickled_code(self) -> None:
        import pickle

        from game_cls.model.builder import build_demo_model
        from game_cls.model.checkpoint_loader import load_model_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            evil = Path(directory) / "evil.pth"
            with evil.open("wb") as stream:
                pickle.dump(
                    {"model": {"__reduce__": (eval, ("1+1",))}},
                    stream,
                )
            model = build_demo_model({"num_classes": 2})
            with self.assertRaisesRegex(RuntimeError, "weights_only"):
                load_model_checkpoint(model, evil)

    def test_legitimate_base_checkpoint_still_loads(self) -> None:
        from game_cls.model.builder import build_demo_model
        from game_cls.model.checkpoint_loader import load_model_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            model = build_demo_model({"num_classes": 2})
            path = Path(directory) / "base.pth"
            torch.save(model.state_dict(), path)
            report = load_model_checkpoint(model, path)
            self.assertGreater(len(report.loaded), 0)

    def test_internal_checkpoint_without_marker_is_refused(self) -> None:
        from game_cls.engine.checkpoint import restore_training_checkpoint
        from game_cls.model.builder import build_demo_model

        with tempfile.TemporaryDirectory() as directory:
            model = build_demo_model({"num_classes": 2})
            path = Path(directory) / "internal.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": {},
                    "global_step": 5,
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "cls_training_checkpoint"):
                restore_training_checkpoint(path, model)

    def test_marker_cannot_smuggle_pickled_code(self) -> None:
        """Audit P0-2: a marker-carrying hostile pickle must not execute.

        The old boundary loaded with weights_only=False and checked the
        ``cls_training_checkpoint`` marker only AFTER unpickling, so a hostile
        payload that also embeds the marker executed before the check. Internal
        checkpoints now load with weights_only=True, so the restricted
        unpickler refuses the payload during deserialization; the marker must
        not weaken that boundary.
        """
        import os
        import pickle

        from game_cls.engine.checkpoint import restore_training_checkpoint
        from game_cls.model.builder import build_demo_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_file = root / "pwned.txt"
            evil = root / "evil_marked.pth"
            with evil.open("wb") as stream:
                pickle.dump(
                    {
                        "cls_training_checkpoint": True,
                        "model": {
                            "__reduce__": (
                                os.system,
                                (f"echo pwned > {marker_file}",),
                            )
                        },
                    },
                    stream,
                )
            model = build_demo_model({"num_classes": 2})
            with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                restore_training_checkpoint(evil, model)
            self.assertFalse(marker_file.exists(), "hostile pickle executed")

    def test_internal_checkpoint_with_marker_restores(self) -> None:
        from game_cls.engine.checkpoint import (
            capture_random_state,
            restore_training_checkpoint,
            save_checkpoint_pair,
        )
        from game_cls.model.builder import build_demo_model

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = build_demo_model({"num_classes": 2})
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            save_checkpoint_pair(
                output,
                "last",
                model,
                optimizer,
                None,
                None,
                1,
                10,
                {},
                {"model": {"checkpoint_path": None}},
                rank_random_states=[capture_random_state()],
            )
            payload = torch.load(
                output / "checkpoint_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(payload.get("cls_training_checkpoint"))
            restored = restore_training_checkpoint(
                output / "checkpoint_last.pth", model
            )
            self.assertEqual(restored["global_step"], 10)


if __name__ == "__main__":
    unittest.main()
