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
    def test_restore_best_does_not_contaminate_checkpoint_last(self) -> None:
        """Audit P0-1: restore_best must not rewrite checkpoint_last.

        checkpoint_last must always hold the true terminal training state.
        Comparing a restore_best=true run against an otherwise-identical
        restore_best=false run proves the selected (best) weights never leak
        into ``last``: both runs save the same model payload, while the best
        checkpoint genuinely differs from the terminal model (so the test would
        catch the old contamination: best weights + terminal optimizer).
        """
        from unittest import mock

        from game_cls.config import load_config
        from game_cls.engine.training.loop import run_training

        def _configure(restore_best: bool, root: Path, name: str) -> dict:
            config = load_config("configs/recipes/example_debug.yaml")
            config["device"]["accelerator"] = "cpu"
            config["experiment"]["output_dir"] = str(root / name)
            config["train"].update(
                {
                    "max_steps": 60,
                    "steps_per_epoch": 30,
                    "local_batch_size": 4,
                    "log_every_steps": 100,
                }
            )
            config["evaluation"].update(
                {
                    "train_probe_every_steps": 0,
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 20,
                    "val_full_at_end": True,
                }
            )
            config["early_stopping"] = {
                "enabled": True,
                "monitor": "selection_score",
                "mode": "max",
                "full_validation_only": True,
                "burn_in_steps": 0,
                "patience_evaluations": 1,
                "min_delta": 0.0,
                "restore_best": restore_best,
            }
            return config

        # First evaluation (step 20) is best; the second (step 40) is worse and
        # exhausts patience, so the run stops at 40 with best frozen at step 20.
        # The terminal model therefore differs from the best checkpoint, which
        # is exactly the condition that used to poison checkpoint_last.
        fake_metrics = [
            {
                "selection_score": 0.9,
                "global_f1_at_decision_threshold": 0.9,
                "cross_entropy": 1.0,
                "worst_game_f1_at_decision_threshold": 0.3,
                "checkpoint_step": 20,
            },
            {
                "selection_score": 0.8,
                "global_f1_at_decision_threshold": 0.8,
                "cross_entropy": 1.2,
                "worst_game_f1_at_decision_threshold": 0.2,
                "checkpoint_step": 40,
            },
        ]

        def _run(restore_best: bool, name: str) -> dict:
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 1)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            config = _configure(restore_best, Path(directory), name)
            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                return run_training(config)

        with tempfile.TemporaryDirectory() as directory:
            with_best = _run(True, "with_best")
            without_best = _run(False, "without_best")
            best_root = Path(with_best["output_dir"])
            plain_root = Path(without_best["output_dir"])

            best_cp = torch.load(
                best_root / "checkpoints" / "checkpoint_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            plain_cp = torch.load(
                plain_root / "checkpoints" / "checkpoint_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            # Same terminal step in both runs.
            self.assertEqual(with_best["global_step"], 40)
            self.assertEqual(without_best["global_step"], 40)
            self.assertEqual(best_cp["global_step"], plain_cp["global_step"])
            # The model payload is identical with and without restore_best:
            # restore_best happened after the save and could not touch it.
            self.assertEqual(set(best_cp["model"]), set(plain_cp["model"]))
            for key in best_cp["model"]:
                self.assertTrue(
                    torch.equal(best_cp["model"][key], plain_cp["model"][key]), key
                )
            # Optimizer / scheduler state are terminal in both (recursive, so
            # tensor payloads are compared with torch.equal, not tensor ==).
            self._assert_flat_state_equal(best_cp["optimizer"], plain_cp["optimizer"])
            self._assert_flat_state_equal(best_cp["scheduler"], plain_cp["scheduler"])
            self.assertEqual(best_cp["sampler_state"], plain_cp["sampler_state"])
            # The best checkpoint genuinely differs from the terminal model, so
            # the test would catch the old contamination.
            best_model = torch.load(
                best_root / "checkpoints" / "model_best_selection.pth",
                map_location="cpu",
                weights_only=True,
            )
            terminal_model = torch.load(
                best_root / "checkpoints" / "model_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(
                any(
                    not torch.equal(best_model[key], terminal_model[key])
                    for key in best_model
                ),
                "test precondition: best must differ from terminal model",
            )
            # The summary still reports the restore for deployment/display.
            summary = json.loads(
                (best_root / "training_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["restored_best"])
            summary_plain = json.loads(
                (plain_root / "training_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary_plain["restored_best"])

    def _assert_flat_state_equal(self, left, right) -> None:
        """Recursively compare optimizer/scheduler state dicts.

        Nested dicts and lists (e.g. optimizer ``state`` / ``param_groups``)
        hold both scalars and tensors; a plain ``==`` on a dict would invoke
        tensor ``==`` and crash on multi-element tensors.
        """
        if isinstance(left, dict) and isinstance(right, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_flat_state_equal(left[key], right[key])
        elif isinstance(left, list) and isinstance(right, list):
            self.assertEqual(len(left), len(right))
            for item_left, item_right in zip(left, right, strict=True):
                self._assert_flat_state_equal(item_left, item_right)
        elif torch.is_tensor(left):
            self.assertTrue(torch.is_tensor(right))
            self.assertTrue(torch.equal(left, right))
        else:
            self.assertEqual(left, right)

    def test_concurrent_resume_of_the_same_run_is_refused(self) -> None:
        """Audit acceptance #6: two processes must not resume the same run.

        A second process resuming while the first holds the run's lock must
        fail loudly instead of loading the same checkpoint and interleaving
        writes.
        """
        from game_cls.config import load_config
        from game_cls.engine.training.loop import run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def _config(resume_dir: Path | None = None) -> dict:
                config = load_config("configs/recipes/example_debug.yaml")
                config["device"]["accelerator"] = "cpu"
                config["experiment"]["output_dir"] = str(resume_dir or root / "run")
                config["train"].update(
                    {
                        "max_steps": 4,
                        "steps_per_epoch": 4,
                        "local_batch_size": 2,
                        "log_every_steps": 100,
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
                config["checkpoint"]["save_last_every_steps"] = 1
                if resume_dir is not None:
                    config["train"]["resume_path"] = str(
                        resume_dir / "checkpoints" / "checkpoint_last.pth"
                    )
                return config

            first = _config()
            first["train"]["stop_after_steps"] = 2
            first_result = run_training(first)
            run_dir = Path(first_result["output_dir"])
            # A second process holds the resume lock; a resume must refuse.
            lock_path = run_dir / ".resume.lock"
            lock_path.write_text("held by another process", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already being resumed"):
                run_training(_config(run_dir))
            # After the lock is released, the same resume succeeds.
            lock_path.unlink()
            result = run_training(_config(run_dir))
            self.assertGreaterEqual(int(result["global_step"]), 4)

    def test_interrupted_resume_matches_uninterrupted_training(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.training.loop import run_training

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
                    "val_quick_every_steps": 0,
                    "val_full_every_steps": 0,
                    "val_full_at_end": False,
                }
            )
            base["checkpoint"]["save_last_every_steps"] = 2

            uninterrupted = copy.deepcopy(base)
            uninterrupted["experiment"]["output_dir"] = str(root / "full")
            full_result = run_training(uninterrupted)

            partial = copy.deepcopy(base)
            partial["train"]["stop_after_steps"] = 2
            partial["experiment"]["output_dir"] = str(root / "resumed")
            partial_result = run_training(partial)
            resumed_run_dir = Path(partial_result["output_dir"])

            resumed = copy.deepcopy(base)
            resumed["experiment"]["output_dir"] = str(resumed_run_dir)
            resumed["train"]["resume_path"] = str(
                resumed_run_dir / "checkpoints" / "checkpoint_last.pth"
            )
            run_training(resumed)

            full_run_dir = Path(full_result["output_dir"])
            full_state = torch.load(
                full_run_dir / "checkpoints" / "model_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                resumed_run_dir / "checkpoints" / "model_last.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(full_state.keys(), resumed_state.keys())
            for key in full_state:
                self.assertTrue(torch.equal(full_state[key], resumed_state[key]), key)
            metric_steps = [
                json.loads(line)["step"]
                for line in (resumed_run_dir / "train_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(metric_steps, [2, 4])


if __name__ == "__main__":
    unittest.main()
