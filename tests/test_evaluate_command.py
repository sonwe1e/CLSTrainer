"""`cls-trainer evaluate` standalone test-split evaluation (step2 plan P0)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def _write_real_png(path: Path, value: int) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((208, 448, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluateCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        # Three disjoint source videos: 01 train, 02 val, 03 test.
        for split, video_id in (("train", "01"), ("val", "02"), ("test", "03")):
            for label in (0, 1):
                for frame_id in (1, 2, 3, 4):
                    _write_real_png(
                        self.root
                        / split
                        / "game_A"
                        / str(label)
                        / f"{video_id}{frame_id:05d}.png",
                        (int(video_id) - 1) * 60 + label * 20 + frame_id,
                    )
        from game_cls.data.image_spec import ImageSpec
        from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
        from game_cls.data.indexing import write_index_bundle

        write_index_bundle(
            self.root / "train",
            self.root / "test",
            self.root / "indexes",
            ImageSpec(width=448, height=208, channels=3),
            ScanPolicy.from_config({}),
            DuplicatePolicy(),
            val_root=self.root / "val",
        )
        # Tiny importable factory used by the run config.
        self.factory_module = self.root / "evalmodel.py"
        self.factory_module.write_text(
            "import torch\n"
            "from torch import nn\n"
            "class Tiny(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            "            nn.Conv2d(6, 2, kernel_size=1, bias=False),\n"
            "            nn.AdaptiveAvgPool2d(1),\n"
            "        )\n"
            "    def forward(self, image0, image1):\n"
            "        return self.net(torch.cat([image0, image1], dim=1)).flatten(1)\n"
            "def build_model(config):\n"
            "    return Tiny()\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(self.root))
        self.addCleanup(lambda: sys.path.remove(str(self.root)))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _run_config(self, output_dir: Path, *, independent_test: bool) -> dict:
        indexes = self.root / "indexes"
        data = {
            "synthetic": False,
            "strict_audit": True,
            "audit_path": str(indexes / "audit.json"),
            "train_index": str(indexes / "train_frames.parquet"),
            "val_index": str(indexes / "val_frames.parquet"),
            "train_video_index": str(indexes / "train_video_entries.parquet"),
            "val_video_index": str(indexes / "val_video_entries.parquet"),
            "backend": "png",
            "width": 448,
            "height": 208,
            "channels": 3,
        }
        if independent_test:
            data["test_index"] = str(indexes / "test_frames.parquet")
            data["test_video_index"] = str(indexes / "test_video_entries.parquet")
        return {
            "experiment": {
                "name": "eval_test",
                "seed": 7,
                "output_dir": str(output_dir),
            },
            "device": {"accelerator": "cpu", "amp": False},
            "data": data,
            "pair": {"test_delta": 2, "train_delta_probability": {2: 1.0}},
            "decision": {"threshold": 0.99},
            "model": {
                "factory": "evalmodel:build_model",
                "checkpoint_path": None,
                "num_classes": 2,
                "trainable_rules": {"head": {"pattern": r"^cls\.", "lr_scale": 1.0}},
            },
            "train": {"local_batch_size": 2},
            "dataloader": {},
            "evaluation": {"amp": False, "amp_dtype": "bfloat16"},
        }

    def test_evaluate_test_split_writes_report_and_summary(self) -> None:
        from game_cls.cli.evaluate import cmd_evaluate

        run_dir = self.root / "run"
        run_dir.mkdir(parents=True)
        config = self._run_config(run_dir, independent_test=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir()
        model = __import__("evalmodel").build_model(config["model"])
        torch.save(model.state_dict(), checkpoints / "model_best_selection.pth")
        args = type(
            "Args",
            (),
            {
                "run": str(run_dir),
                "runs_root": str(self.root / "runs"),
                "checkpoint": "best_selection",
                "split": "test",
                "config": None,
            },
        )()
        code = cmd_evaluate(args)
        self.assertEqual(code, 0)
        summary = json.loads(
            (run_dir / "test_evaluation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["split"], "test")
        self.assertIn("selection_score", summary)
        self.assertIn("threshold_loss", summary)
        self.assertIn("positive_margin_pass_rate", summary)
        self.assertIn("negative_margin_p90", summary)
        self.assertEqual(summary["sample_count"], 4)
        history = [
            json.loads(line)
            for line in (run_dir / "metrics" / "evaluation.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(history[-1]["split"], "test")
        self.assertEqual(history[-1]["scope"], "full")
        self.assertTrue(list((run_dir / "reports").glob("test_full_*")))

    def test_evaluate_test_split_refused_without_independent_test(self) -> None:
        from game_cls.cli.evaluate import cmd_evaluate

        run_dir = self.root / "run_without_test"
        run_dir.mkdir(parents=True)
        config = self._run_config(run_dir, independent_test=False)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir()
        model = __import__("evalmodel").build_model(config["model"])
        torch.save(model.state_dict(), checkpoints / "model_best_selection.pth")
        args = type(
            "Args",
            (),
            {
                "run": str(run_dir),
                "runs_root": str(self.root / "runs"),
                "checkpoint": "best_selection",
                "split": "test",
                "config": None,
            },
        )()
        code = cmd_evaluate(args)
        self.assertEqual(code, 3)
        self.assertFalse((run_dir / "test_evaluation.json").exists())

    def test_evaluate_validation_works_without_test_split(self) -> None:
        from game_cls.cli.evaluate import cmd_evaluate

        run_dir = self.root / "run_without_test"
        run_dir.mkdir(parents=True)
        config = self._run_config(run_dir, independent_test=False)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir()
        model = __import__("evalmodel").build_model(config["model"])
        torch.save(model.state_dict(), checkpoints / "model_best_selection.pth")
        args = type(
            "Args",
            (),
            {
                "run": str(run_dir),
                "runs_root": str(self.root / "runs"),
                "checkpoint": "best_selection",
                "split": "validation",
                "config": None,
            },
        )()
        code = cmd_evaluate(args)
        self.assertEqual(code, 0)
        self.assertTrue((run_dir / "validation_evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
