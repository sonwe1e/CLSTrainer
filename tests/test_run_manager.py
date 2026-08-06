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


def _fast_training_overrides(output_dir: str) -> dict:
    return {
        "device": {"accelerator": "cpu"},
        "experiment": {"output_dir": output_dir},
        "train": {
            "max_steps": 2,
            "steps_per_epoch": 2,
            "local_batch_size": 2,
            "log_every_steps": 1,
        },
        "evaluation": {
            "quick_test_every_steps": 0,
            "full_test_every_steps": 0,
            "full_test_at_end": False,
        },
        "checkpoint": {"save_last_every_steps": 0},
    }


def _apply(container: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(container.get(key), dict):
            _apply(container[key], value)
        else:
            container[key] = value
    return container


@unittest.skipIf(torch is None, "torch is not installed")
class UniqueRunDirectoryTests(unittest.TestCase):
    def test_two_starts_create_two_disjoint_runs(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs_root"
            created = []
            for _ in range(2):
                config = load_config("configs/cuda_debug.yaml")
                _apply(config, _fast_training_overrides(str(root)))
                config["experiment"]["run_mode"] = "unique"
                result = run_training(config)
                created.append(Path(result["output_dir"]))
            self.assertNotEqual(created[0], created[1])
            for run_dir in created:
                self.assertTrue(run_dir.is_dir())
                self.assertTrue((run_dir / "manifest.json").is_file())
                self.assertTrue((run_dir / "status.json").is_file())
                self.assertTrue((run_dir / "summary.md").is_file())
                self.assertTrue(
                    (run_dir / "resolved_config.json").is_file()
                )
                status = json.loads(
                    (run_dir / "status.json").read_text(encoding="utf-8")
                )
                self.assertEqual(status["state"], "SUCCEEDED")
            # The runs root itself holds an append-only index: one RUNNING
            # record per start plus one terminal record per finish. Readers
            # aggregate by run identity, so the two runs surface as two
            # records with final states.
            index_lines = (
                (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(len(index_lines), 4)
            records = [json.loads(line) for line in index_lines]
            self.assertEqual(
                {record["state"] for record in records}, {"RUNNING", "SUCCEEDED"}
            )
            from game_cls.cli import _find_index_records

            aggregated = _find_index_records(root)
            self.assertEqual(len(aggregated), 2)
            self.assertEqual(
                {record["state"] for record in aggregated}, {"SUCCEEDED"}
            )

    def test_fixed_mode_keeps_legacy_in_place_behavior(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixed_run"
            config = load_config("configs/cuda_debug.yaml")
            _apply(config, _fast_training_overrides(str(output)))
            result = run_training(config)
            self.assertEqual(Path(result["output_dir"]), output)
            self.assertTrue((output / "train_metrics.jsonl").is_file())

    def test_failed_run_records_status_and_failure_log(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs_root"
            config = load_config("configs/cuda_debug.yaml")
            _apply(config, _fast_training_overrides(str(root)))
            config["experiment"]["run_mode"] = "unique"
            config["train"]["resume_path"] = str(root / "does_not_exist.pth")
            with self.assertRaises(FileNotFoundError):
                run_training(config)
            (run_dir,) = [
                path
                for date_dir in root.iterdir()
                if date_dir.is_dir()
                for path in date_dir.iterdir()
            ]
            status = json.loads(
                (run_dir / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "FAILED")
            self.assertEqual(status["error_type"], "FileNotFoundError")
            self.assertTrue((run_dir / "failure.log").is_file())
            records = [
                json.loads(line)
                for line in (root / "index.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(records[-1]["state"], "FAILED")

    def test_resume_reuses_the_same_run_directory(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs_root"
            config = load_config("configs/cuda_debug.yaml")
            _apply(config, _fast_training_overrides(str(root)))
            config["experiment"]["run_mode"] = "unique"
            config["checkpoint"]["save_last_every_steps"] = 1
            first = run_training(config)
            run_dir = Path(first["output_dir"])

            resumed = load_config("configs/cuda_debug.yaml")
            _apply(resumed, _fast_training_overrides(str(root)))
            resumed["experiment"]["run_mode"] = "fixed"
            resumed["experiment"]["output_dir"] = str(run_dir)
            resumed["train"]["max_steps"] = 4
            resumed["train"]["resume_path"] = str(
                run_dir / "checkpoints" / "checkpoint_last.pth"
            )
            second = run_training(resumed)
            self.assertEqual(Path(second["output_dir"]), run_dir)
            self.assertEqual(second["global_step"], 4)
            # Resume must not allocate another run directory.
            run_dirs = [
                path
                for date_dir in root.iterdir()
                if date_dir.is_dir()
                for path in date_dir.iterdir()
                if path.is_dir()
            ]
            self.assertEqual(run_dirs, [run_dir])


class AllocateRunDirTests(unittest.TestCase):
    def test_allocation_never_overwrites(self) -> None:
        from game_cls.runs import allocate_run_dir

        with tempfile.TemporaryDirectory() as directory:
            first, first_id = allocate_run_dir(directory, "My Run!")
            second, second_id = allocate_run_dir(directory, "My Run!")
            self.assertNotEqual(first, second)
            self.assertNotEqual(first_id, second_id)
            self.assertTrue(first.is_dir())
            self.assertIn("my-run", first.name)

    def test_status_updates_are_merged(self) -> None:
        from game_cls.runs import update_status

        with tempfile.TemporaryDirectory() as directory:
            update_status(directory, state="RUNNING", step=1)
            payload = update_status(directory, state="SUCCEEDED", step=2)
            self.assertEqual(payload["state"], "SUCCEEDED")
            self.assertEqual(payload["step"], 2)
            self.assertIn("last_update", payload)


class CliWorkflowTests(unittest.TestCase):
    def _run_cli(self, *args: str, expect_failure: bool = False) -> str:
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
        if expect_failure:
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
        else:
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
        return completed.stdout + completed.stderr

    def test_dry_run_prints_plan_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self._run_cli(
                "train",
                "--config",
                "configs/cuda_debug.yaml",
                "--dry-run",
                f"experiment.output_dir={directory}/root",
            )
            self.assertIn("DRY RUN", output)
            self.assertIn("total steps        : 100", output)
            self.assertIn("decision threshold : 0.99", output)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_typo_override_fails_with_suggestion(self) -> None:
        output = self._run_cli(
            "config",
            "validate",
            "--config",
            "configs/cuda_debug.yaml",
            "optimzier.learning_rate=0.0001",
            expect_failure=True,
        )
        self.assertIn("Did you mean 'optimizer.learning_rate'", output)

    def test_config_reference_lists_decision_threshold(self) -> None:
        output = self._run_cli("config", "reference")
        self.assertIn("decision.threshold", output)
        self.assertNotIn("optimizer.name", output)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_train_cli_creates_unique_run_and_concise_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs_root"
            output = self._run_cli(
                "train",
                "--config",
                "configs/cuda_debug.yaml",
                "device.accelerator=cpu",
                f"experiment.output_dir={root}",
                "train.max_steps=2",
                "train.steps_per_epoch=2",
                "train.log_every_steps=1",
                "evaluation.quick_test_every_steps=0",
                "evaluation.full_test_every_steps=0",
                "evaluation.full_test_at_end=false",
                "checkpoint.save_last_every_steps=0",
            )
            self.assertIn("=== Training finished ===", output)
            self.assertNotIn("'evaluation_state'", output)
            run_dirs = [
                path
                for date_dir in root.iterdir()
                if date_dir.is_dir()
                for path in date_dir.iterdir()
                if path.is_dir()
            ]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue((run_dirs[0] / "console.log").is_file())
            console = (run_dirs[0] / "console.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("Trainable parameters", console)
            # run list / run show observe the recorded run.
            listing = self._run_cli("run", "list", "--root", str(root))
            self.assertIn("SUCCEEDED", listing)
            shown = self._run_cli(
                "run", "show", "latest", "--root", str(root)
            )
            self.assertIn("state        : SUCCEEDED", shown)


if __name__ == "__main__":
    unittest.main()
