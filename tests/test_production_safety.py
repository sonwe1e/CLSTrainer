from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

from game_cls.config import load_config
from game_cls.engine.trainer import validate_training_config


class ProductionConfigTests(unittest.TestCase):
    def test_npu_config_cannot_run_with_placeholder_model(self) -> None:
        config = load_config("configs/npu_1p.yaml")
        with self.assertRaisesRegex(RuntimeError, "placeholder"):
            validate_training_config(config)

    def test_real_factory_and_existing_checkpoint_pass_gate(self) -> None:
        config = load_config("configs/npu_1p.yaml")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base.pth"
            checkpoint.write_bytes(b"checkpoint")
            config["model"]["factory"] = "real_model.factory:build"
            config["model"]["checkpoint_path"] = str(checkpoint)
            validate_training_config(config)


@unittest.skipIf(torch is None, "torch is not installed")
class ProductionCheckpointTests(unittest.TestCase):
    def test_non_cls_missing_weights_are_fatal(self) -> None:
        from game_cls.model.builder import build_demo_model
        from game_cls.model.checkpoint_loader import (
            LoadReport,
            validate_production_load,
        )

        model = build_demo_model({})
        report = LoadReport(
            loaded=("cls.weight", "cls.bias"),
            missing=("backbone.0.weight",),
            unexpected=(),
            shape_mismatch=(),
        )
        with self.assertRaisesRegex(RuntimeError, "frozen backbone"):
            validate_production_load(model, report)

    def test_trainable_only_checkpoint_restores_on_top_of_base(self) -> None:
        from game_cls.engine.checkpoint import (
            restore_training_checkpoint,
            save_checkpoint_pair,
        )
        from game_cls.model.builder import build_demo_model
        from game_cls.model.freeze_policy import configure_trainable_parameters

        model = build_demo_model({})
        configure_trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad]
        )
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint_pair(
                directory,
                "last",
                model,
                optimizer,
                None,
                None,
                epoch=0,
                global_step=1,
                best_metrics={},
                config={"model": {"checkpoint_path": None}},
                state_mode="trainable_only",
                write_model_only=False,
            )
            payload = torch.load(
                Path(directory) / "checkpoint_last.pth",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["model_state_mode"], "trainable_only")
            self.assertEqual(set(payload["model"]), {"cls.weight", "cls.bias"})
            fresh = build_demo_model({})
            restore_training_checkpoint(
                Path(directory) / "checkpoint_last.pth", fresh
            )


if __name__ == "__main__":
    unittest.main()
