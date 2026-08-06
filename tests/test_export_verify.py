"""Model export + release gate (step5 P6)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ExportVerifyTests(unittest.TestCase):
    def _trained_run(self, directory: str) -> Path:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        config = load_config("configs/cuda_debug.yaml")
        config["experiment"]["output_dir"] = str(Path(directory) / "run")
        config["train"].update(
            {
                "max_steps": 8,
                "steps_per_epoch": 8,
                "local_batch_size": 4,
                "log_every_steps": 8,
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
        config["checkpoint"].update({"save_topk": 0})
        config.pop("early_stopping", None)
        result = run_training(config)
        return Path(result["output_dir"])

    def test_weights_export_roundtrip(self) -> None:
        import torch

        from game_cls.cli.export import cmd_export
        from game_cls.model.builder import build_model

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._trained_run(directory)
            export_dir = Path(directory) / "exports"
            args = type(
                "Args",
                (),
                {
                    "run": str(run_dir),
                    "runs_root": str(Path(directory) / "runs"),
                    "config": None,
                    "checkpoint": "last",
                    "format": "weights",
                    "out": str(export_dir),
                },
            )()
            rc = cmd_export(args)
            self.assertEqual(rc, 0)
            weights = export_dir / "model_last.pth"
            manifest = json.loads(
                (export_dir / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(weights.is_file())
            self.assertEqual(manifest["decision.threshold"], 0.99)
            self.assertEqual(manifest["shape"], [1, 2, 3, 208, 448])
            # The exported state dict round-trips into a fresh model.
            model = build_model(
                {"factory": "game_cls.model.builder:build_demo_model", "num_classes": 2}
            )
            model.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True),
                strict=True,
            )


class ReleaseGateTests(unittest.TestCase):
    def test_release_gate_fails_on_template(self) -> None:
        from game_cls.cli.config_tools import cmd_config_validate

        args = type(
            "Args",
            (),
            {
                "config": "configs/recipes/game_cls_production.yaml",
                "overrides": [],
                "release": True,
            },
        )()
        self.assertEqual(cmd_config_validate(args), 2)

    def test_release_gate_passes_with_baseline(self) -> None:
        import tempfile

        import yaml

        from game_cls.cli.config_tools import cmd_config_validate

        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "release.yaml"
            config = yaml.safe_load(
                Path("configs/recipes/game_cls_production.yaml").read_text(
                    encoding="utf-8"
                )
            )
            config.setdefault("evaluation", {})["minimum_worst_game_f1"] = 0.75
            config.setdefault("benchmark", {})["gate_metrics"] = {
                "global_fpr_at_decision_threshold": 0.01
            }
            cfg_path.write_text(
                yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
            )
            args = type(
                "Args",
                (),
                {"config": str(cfg_path), "overrides": [], "release": True},
            )()
            self.assertEqual(cmd_config_validate(args), 0)


if __name__ == "__main__":
    unittest.main()
