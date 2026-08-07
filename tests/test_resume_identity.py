"""Resume preserves Run identity and index consistency (step3 P0-2)."""

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
class ResumeIdentityTests(unittest.TestCase):
    def _config(self) -> dict:
        from game_cls.config import load_config

        config = load_config("configs/recipes/example_debug.yaml")
        config["device"]["accelerator"] = "cpu"
        config["train"].update(
            {
                "max_steps": 4,
                "steps_per_epoch": 4,
                "local_batch_size": 2,
                "log_every_steps": 100,
                "verify_frozen_parameters": False,
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
        config["checkpoint"]["save_last_every_steps"] = 2
        config.pop("early_stopping", None)
        return config

    def test_resume_keeps_manifest_run_id_and_logs_resume_event(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            first = self._config()
            first["experiment"]["run_mode"] = "unique"
            first["experiment"]["output_dir"] = str(root)
            first["train"]["stop_after_steps"] = 2
            result1 = run_training(first)
            run_dir = Path(result1["output_dir"])
            run_id = result1["run_id"]
            self.assertIsNotNone(run_id)

            manifest1 = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            status1 = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            initial_config = json.loads(
                (run_dir / "resolved_config.initial.json").read_text(encoding="utf-8")
            )

            # Second run resumes into the SAME directory, as
            # `cls-trainer train --resume <run_dir>` does.
            resumed = self._config()
            resumed["experiment"]["output_dir"] = str(run_dir)
            resumed["experiment"]["run_mode"] = "fixed"
            resumed["train"]["resume_path"] = str(
                run_dir / "checkpoints" / "checkpoint_last.pth"
            )
            result2 = run_training(
                resumed,
                run_meta={"runs_root": str(root), "resume_type": "exact"},
            )

            # The Run identity must be preserved end to end.
            self.assertEqual(result2["run_id"], run_id)
            manifest2 = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest2["run_id"], run_id)
            self.assertEqual(manifest2["created"], manifest1["created"])
            self.assertEqual(manifest2["command"], manifest1["command"])

            # The first resolved config is never touched by the resume.
            self.assertEqual(
                json.loads(
                    (run_dir / "resolved_config.initial.json").read_text(
                        encoding="utf-8"
                    )
                ),
                initial_config,
            )
            self.assertEqual(initial_config["experiment"]["run_mode"], "unique")

            # status.json keeps the original `started` and records
            # `resumed_at` instead of overwriting it.
            status2 = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status2["started"], status1["started"])
            self.assertTrue(status2.get("resumed_at"))
            self.assertEqual(status2["state"], "SUCCEEDED")

            # The resume event log records the continuation.
            events = [
                json.loads(line)
                for line in (run_dir / "resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(
                events[0]["checkpoint"],
                str(run_dir / "checkpoints" / "checkpoint_last.pth"),
            )
            self.assertIn("command", events[0])
            self.assertIn("resume_type", events[0])

            # The append-only index is aggregated per run identity: the
            # pre-resume RUNNING/FAILED states must not shadow the final one.
            from game_cls.cli import _find_index_records

            records = _find_index_records(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_id"], run_id)
            self.assertEqual(records[0]["state"], "SUCCEEDED")
            self.assertEqual(records[0]["global_step"], 4)
            self.assertEqual(
                Path(records[0]["output_dir"]).resolve(), run_dir.resolve()
            )

    def test_resume_without_prior_manifest_creates_one(self) -> None:
        """A plain fixed-mode re-run of an existing dir must not clobber
        resolved_config.json without keeping the initial snapshot."""
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            first = self._config()
            first["experiment"]["output_dir"] = str(root / "run")
            first["train"]["stop_after_steps"] = 2
            run_training(first)

            second = self._config()
            second["experiment"]["output_dir"] = str(root / "run")
            second["train"]["resume_path"] = str(
                root / "run" / "checkpoints" / "checkpoint_last.pth"
            )
            run_training(second)

            run_dir = root / "run"
            self.assertTrue((run_dir / "resolved_config.initial.json").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "resume_events.jsonl").is_file())
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "SUCCEEDED")

    def test_resume_type_extends_when_max_steps_grows(self) -> None:
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
                }
            )
            base["evaluation"].update(
                {
                    "train_probe_every_steps": 0,
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 0,
                    "val_full_at_end": False,
                }
            )
            base["checkpoint"]["save_last_every_steps"] = 2
            base["experiment"]["output_dir"] = str(root / "run")
            base["train"]["stop_after_steps"] = 2
            run_training(base)

            extended = copy.deepcopy(base)
            extended["train"].pop("stop_after_steps")
            extended["train"]["max_steps"] = 6
            extended["train"]["resume_path"] = str(
                root / "run" / "checkpoints" / "checkpoint_last.pth"
            )
            run_training(
                extended,
                run_meta={"runs_root": str(root), "resume_type": "extend"},
            )
            events = [
                json.loads(line)
                for line in (root / "run" / "resume_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
