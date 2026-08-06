"""Top-K checkpoint registry (step3 P0-1b).

``ConstrainedTopKTests`` covers step6: with ``topk_monitor: selection_score``
the registry must order by the selection rank key and must never rank an
ineligible checkpoint above an eligible one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


def _constrained_cfg(**overrides) -> dict:
    cfg = {
        "selection_mode": "constrained",
        "max_global_fpr": 0.01,
        "max_worst_game_fpr": 0.02,
        "min_positive_recall": 0.8,
        "minimum_worst_game_f1": None,
    }
    cfg.update(overrides)
    return cfg


def _metrics(
    step: int,
    *,
    global_fpr: float = 0.005,
    global_recall: float = 0.9,
    worst_game_recall: float = 0.8,
    negative_p999: float = -2.0,
    cross_entropy: float = 1.0,
) -> dict:
    return {
        "global_fpr_at_decision_threshold": global_fpr,
        "worst_game_fpr_at_decision_threshold": 0.01,
        "global_positive_recall_at_decision_threshold": global_recall,
        "worst_game_positive_recall_at_decision_threshold": worst_game_recall,
        "negative_score_p999": negative_p999,
        "cross_entropy": cross_entropy,
        # The scalar the old code sorted by: constant across the fixture, so
        # a scalar sort produces insertion order and proves nothing.
        "selection_score": 0.9,
        "checkpoint_step": step,
    }


@unittest.skipIf(torch is None, "torch is not installed")
class TopKCheckpointTests(unittest.TestCase):
    def _config(self, directory: str, save_topk: int) -> dict:
        from game_cls.config import load_config

        config = load_config("configs/cuda_debug.yaml")
        config["experiment"]["output_dir"] = str(Path(directory) / "run")
        config["train"].update(
            {
                "max_steps": 60,
                "steps_per_epoch": 30,
                "local_batch_size": 4,
                "log_every_steps": 30,
            }
        )
        config["evaluation"].update(
            {
                "train_probe_every_steps": 0,
                "val_quick_every_steps": 0,
                "val_full_every_steps": 20,
                "val_full_at_end": False,
            }
        )
        config["checkpoint"].update(
            {
                "save_topk": save_topk,
                "topk_monitor": "selection_score",
            }
        )
        config.pop("early_stopping", None)
        return config

    def test_keeps_two_best_and_evicts_worst(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, save_topk=2)
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.9,
                    "global_f1_tau099": 0.9,
                    "cross_entropy": 0.8,
                    "worst_game_f1_tau099": 0.7,
                    "checkpoint_step": 40,
                },
                {
                    "selection_score": 0.4,
                    "global_f1_tau099": 0.4,
                    "cross_entropy": 1.2,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 60,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 2)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)

            registry = result["evaluation_state"]["topk_registry"]
            # Sorted best-first: step 40 (0.9), step 20 (0.5); step 60 (0.4)
            # must have been evicted.
            self.assertEqual([entry["step"] for entry in registry], [40, 20])
            self.assertEqual(registry[0]["value"], 0.9)
            self.assertEqual(registry[1]["value"], 0.5)

            checkpoints = Path(directory) / "run" / "checkpoints"
            for step in (20, 40):
                self.assertTrue(
                    (checkpoints / f"model_topk_{step:08d}.pth").is_file(),
                    f"missing topk file for step {step}",
                )
                self.assertTrue(
                    (checkpoints / f"checkpoint_topk_{step:08d}.pth").is_file()
                )
            # The evicted checkpoint must be gone.
            self.assertFalse((checkpoints / "model_topk_00000060.pth").exists())
            # Registry persisted next to the checkpoints.
            registry_file = checkpoints / "topk_registry.json"
            self.assertTrue(registry_file.is_file())
            persisted = json.loads(registry_file.read_text(encoding="utf-8"))
            self.assertEqual(persisted["monitor"], "selection_score")
            self.assertEqual(
                [entry["step"] for entry in persisted["entries"]], [40, 20]
            )
            # Summary payload carries the list for run show / summary.md.
            summary = json.loads(
                (Path(directory) / "run" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [entry["step"] for entry in summary["topk_checkpoints"]],
                [40, 20],
            )

    def test_lower_better_monitor_keeps_lowest_cross_entropy(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, save_topk=2)
            config["checkpoint"]["topk_monitor"] = "cross_entropy"
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.9,
                    "global_f1_tau099": 0.9,
                    "cross_entropy": 0.6,
                    "worst_game_f1_tau099": 0.7,
                    "checkpoint_step": 40,
                },
                {
                    "selection_score": 0.4,
                    "global_f1_tau099": 0.4,
                    "cross_entropy": 0.8,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 60,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 2)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)

            registry = result["evaluation_state"]["topk_registry"]
            # Lowest CE first: step 40 (0.6), then step 60 (0.8); step 20 (1.0)
            # evicted.
            self.assertEqual([entry["step"] for entry in registry], [40, 60])
            checkpoints = Path(directory) / "run" / "checkpoints"
            self.assertTrue((checkpoints / "model_topk_00000040.pth").is_file())
            self.assertTrue((checkpoints / "model_topk_00000060.pth").is_file())
            self.assertFalse((checkpoints / "model_topk_00000020.pth").exists())

    def test_disabled_topk_writes_no_topk_files(self) -> None:
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, save_topk=0)
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                }
            ]

            def fake_run_evaluation(**kwargs):
                return mock.Mock(metrics=dict(fake_metrics[0]))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)
            self.assertEqual(result["evaluation_state"]["topk_registry"], [])
            checkpoints = Path(directory) / "run" / "checkpoints"
            self.assertFalse(list(checkpoints.glob("model_topk_*.pth")))


@unittest.skipIf(torch is None, "torch is not installed")
class ConstrainedTopKTests(unittest.TestCase):
    def _run(self, directory: str, payloads: list[dict], *, save_topk: int) -> dict:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        config = load_config("configs/cuda_debug.yaml")
        config["experiment"]["output_dir"] = str(Path(directory) / "run")
        config["train"].update(
            {
                "max_steps": 20 * len(payloads),
                "steps_per_epoch": 20 * len(payloads),
                "local_batch_size": 4,
                "log_every_steps": 20,
            }
        )
        config["evaluation"].update(
            {
                "train_probe_every_steps": 0,
                "val_quick_every_steps": 0,
                "val_full_every_steps": 20,
                "val_full_at_end": False,
                **_constrained_cfg(),
            }
        )
        config["checkpoint"].update(
            {"save_topk": save_topk, "topk_monitor": "selection_score"}
        )
        config.pop("early_stopping", None)
        calls = {"index": 0}

        def fake_run_evaluation(**kwargs):
            payload = payloads[min(calls["index"], len(payloads) - 1)]
            calls["index"] += 1
            return mock.Mock(metrics=dict(payload))

        with mock.patch(
            "game_cls.engine.training.loop._run_evaluation",
            side_effect=fake_run_evaluation,
        ):
            return run_training(config)

    def test_order_follows_the_rank_key_not_the_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # Identical global recall / selection_score everywhere: only the
            # worst-game-recall and negative-p99.9 tie-breakers separate them,
            # so a scalar sort would keep insertion order [20, 40, 60].
            payloads = [
                _metrics(20, worst_game_recall=0.80, negative_p999=-2.0),
                _metrics(40, worst_game_recall=0.95, negative_p999=-2.0),
                _metrics(60, worst_game_recall=0.80, negative_p999=-4.0),
            ]
            result = self._run(directory, payloads, save_topk=3)
            registry = result["evaluation_state"]["topk_registry"]
            # 40 wins on worst-game recall; 60 beats 20 on the negative tail.
            self.assertEqual([entry["step"] for entry in registry], [40, 60, 20])
            self.assertEqual(
                [entry["sort_value"] for entry in registry],
                [[1.0, 0.9, 0.95, 2.0], [1.0, 0.9, 0.8, 4.0], [1.0, 0.9, 0.8, 2.0]],
            )
            # The reported value stays the primary objective, so `run show`
            # and summary.md keep printing one comparable number.
            self.assertEqual([entry["value"] for entry in registry], [0.9, 0.9, 0.9])

    def test_ineligible_never_outranks_eligible_and_is_evicted_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # Step 40 has the best raw recall of the three but fails the
            # global-FPR gate, so it must rank last despite winning the scalar.
            payloads = [
                _metrics(20, global_recall=0.85),
                _metrics(40, global_fpr=0.9, global_recall=0.99),
                _metrics(60, global_recall=0.82),
            ]
            result = self._run(directory, payloads, save_topk=2)
            registry = result["evaluation_state"]["topk_registry"]
            self.assertEqual([entry["step"] for entry in registry], [20, 60])
            self.assertEqual([entry["eligible"] for entry in registry], [True, True])
            checkpoints = Path(directory) / "run" / "checkpoints"
            # The ineligible snapshot was admitted (it was the only candidate
            # at the time) but evicted once two eligible ones existed.
            self.assertFalse((checkpoints / "model_topk_00000040.pth").exists())
            for step in (20, 60):
                self.assertTrue(
                    (checkpoints / f"model_topk_{step:08d}.pth").is_file(),
                    f"missing topk file for step {step}",
                )

    def test_ineligible_is_kept_when_nothing_is_eligible_yet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payloads = [
                _metrics(20, global_fpr=0.9, global_recall=0.95),
                _metrics(40, global_fpr=0.9, global_recall=0.99),
            ]
            result = self._run(directory, payloads, save_topk=2)
            registry = result["evaluation_state"]["topk_registry"]
            # Ranked among themselves by the rank key, all flagged ineligible,
            # so a constrained run still has snapshots to inspect early on.
            self.assertEqual([entry["step"] for entry in registry], [40, 20])
            self.assertEqual([entry["eligible"] for entry in registry], [False, False])

    def test_registry_survives_a_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payloads = [
                _metrics(20, worst_game_recall=0.80),
                _metrics(40, worst_game_recall=0.95),
            ]
            self._run(directory, payloads, save_topk=2)
            path = Path(directory) / "run" / "checkpoints" / "topk_registry.json"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["monitor"], "selection_score")
            self.assertEqual(
                [entry["step"] for entry in persisted["entries"]], [40, 20]
            )
            # A list decodes back as a list and still compares; a tuple would
            # not survive, which is why the key is normalized.
            keys = [entry["sort_value"] for entry in persisted["entries"]]
            self.assertEqual(keys, sorted(keys, reverse=True))
            self.assertIsInstance(keys[0], list)


class TopKEntrySortValueTests(unittest.TestCase):
    def test_legacy_entries_without_sort_value_still_rank(self) -> None:
        from game_cls.engine.training.run_io import _topk_entry_sort_value

        # Entries restored from a checkpoint written before step6.
        self.assertEqual(
            _topk_entry_sort_value({"value": 0.7}, lower_better=False), [0.7]
        )
        self.assertEqual(
            _topk_entry_sort_value({"value": 0.7}, lower_better=True), [-0.7]
        )
        # A present sort_value wins over the scalar.
        self.assertEqual(
            _topk_entry_sort_value(
                {"value": 0.7, "sort_value": [1.0, 0.9, 0.8]}, lower_better=False
            ),
            [1.0, 0.9, 0.8],
        )


if __name__ == "__main__":
    unittest.main()
