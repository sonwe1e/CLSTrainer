from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:
    torch = None


def _cli(*args: str, expect_code: int = 0) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    completed = subprocess.run(
        [sys.executable, "-m", "game_cls.cli", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if expect_code is not None:
        assert completed.returncode == expect_code, (
            completed.stdout + completed.stderr
        )
    return completed.returncode, completed.stdout + completed.stderr


def _base_overrides(root: str) -> list[str]:
    return [
        "device.accelerator=cpu",
        f"experiment.output_dir={root}",
        "train.max_steps=2",
        "train.steps_per_epoch=2",
        "train.log_every_steps=1",
        "evaluation.quick_test_every_steps=0",
        "evaluation.full_test_every_steps=0",
        "evaluation.full_test_at_end=false",
        "checkpoint.save_last_every_steps=1",
    ]


def _run_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for date_dir in root.iterdir()
        if date_dir.is_dir()
        for path in date_dir.iterdir()
        if path.is_dir()
    )


@unittest.skipIf(torch is None, "torch is not installed")
class RunToolsTests(unittest.TestCase):
    def test_overview_compare_fork_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs_root"
            _cli(
                "train",
                "--config",
                "configs/cuda_debug.yaml",
                *_base_overrides(str(root)),
            )
            (run_a,) = _run_dirs(root)

            overview = run_a / "overview.html"
            self.assertTrue(overview.is_file())
            html = overview.read_text(encoding="utf-8")
            self.assertIn("SUCCEEDED", html)
            self.assertIn("<svg", html)
            self.assertIn("Loss", html)

            # Fork run A with a one-knob change; lineage must be recorded.
            _, fork_output = _cli(
                "train",
                "--fork",
                str(run_a),
                "optimizer.learning_rate=0.01",
                "train.max_steps=2",
                "train.steps_per_epoch=2",
                "train.log_every_steps=1",
                "evaluation.quick_test_every_steps=0",
                "evaluation.full_test_every_steps=0",
                "evaluation.full_test_at_end=false",
                "checkpoint.save_last_every_steps=0",
            )
            self.assertIn("Training finished", fork_output)
            run_b = [
                path for path in _run_dirs(root) if path != run_a
            ][0]
            manifest_a = json.loads(
                (run_a / "manifest.json").read_text(encoding="utf-8")
            )
            manifest_b = json.loads(
                (run_b / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest_b.get("parent_run_id"), manifest_a.get("run_id")
            )
            self.assertEqual(manifest_b.get("forked_from"), str(run_a))
            config_b = json.loads(
                (run_b / "resolved_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config_b["optimizer"]["learning_rate"], 0.01)
            self.assertIsNone(config_b["train"]["resume_path"])

            # compare reports the config diff and both runs.
            _, compare_output = _cli(
                "run", "compare", str(run_a), str(run_b), "--root", str(root)
            )
            self.assertIn("optimizer.learning_rate", compare_output)
            self.assertIn("0.001", compare_output)
            self.assertIn("0.01", compare_output)
            self.assertIn("Config differences", compare_output)

            # run id also resolves targets.
            _, compare_by_id = _cli(
                "run",
                "compare",
                manifest_a["run_id"],
                manifest_b["run_id"],
                "--root",
                str(root),
            )
            self.assertIn("optimizer.learning_rate", compare_by_id)

            # TensorBoard export (skipped when tensorboard is unavailable).
            try:
                import torch.utils.tensorboard  # noqa: F401
            except ImportError:
                return
            tb_out = Path(directory) / "tb"
            _, export_output = _cli(
                "run",
                "export-tensorboard",
                str(run_a),
                "--out",
                str(tb_out),
            )
            self.assertIn("TensorBoard events written", export_output)
            self.assertTrue(list(tb_out.glob("events.out.tfevents.*")))

    def test_resume_refuses_critical_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs_root"
            _cli(
                "train",
                "--config",
                "configs/cuda_debug.yaml",
                *_base_overrides(str(root)),
            )
            (run_a,) = _run_dirs(root)

            code, output = _cli(
                "train",
                "--resume",
                str(run_a),
                "decision.threshold=0.5",
                "train.max_steps=4",
                expect_code=None,
            )
            self.assertEqual(code, 3, output)
            self.assertIn("critical config drift", output)
            self.assertIn("decision.threshold", output)

            # Non-critical drift warns but proceeds.
            code, output = _cli(
                "train",
                "--resume",
                str(run_a),
                "optimizer.learning_rate=0.02",
                "train.max_steps=4",
                expect_code=None,
            )
            self.assertEqual(code, 0, output)
            self.assertIn("[WARNING] resume config drift", output)
            self.assertIn("optimizer.learning_rate", output)
            self.assertIn("Training finished", output)
            status = json.loads(
                (run_a / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "SUCCEEDED")
            self.assertEqual(status["step"], 4)


if __name__ == "__main__":
    unittest.main()
