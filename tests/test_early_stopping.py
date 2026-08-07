"""Early stopping and multi-objective checkpoints (step2 plan P2).

``EarlyStoppingSelectionContractTests`` covers step6: early stopping must
judge improvement with the same eligibility + rank key as best-checkpoint
selection, instead of reading one scalar monitor.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from game_cls.engine.training.early_stopping import (
    _early_stopping_defaults,
    _early_stopping_monitor_label,
    _update_early_stopping,
)

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
        # The scalar the old code monitored: deliberately constant, so a test
        # that still reads it cannot tell these payloads apart.
        "selection_score": 0.9,
    }


def _early_cfg(**overrides) -> dict:
    cfg = {
        "enabled": True,
        "monitor": "selection_score",
        "mode": "max",
        "burn_in_steps": 0,
        "patience_evaluations": 1,
        "min_delta": 0.0,
    }
    cfg.update(overrides)
    return cfg


class EarlyStoppingSelectionContractTests(unittest.TestCase):
    def test_tie_breaker_gain_is_an_improvement_not_a_plateau(self) -> None:
        # Same global recall (so the scalar selection_score is identical),
        # better worst-game recall. Selection calls this an improvement, so
        # early stopping must too.
        state = _early_stopping_defaults()
        cfg, evaluation = _early_cfg(), _constrained_cfg()
        first = _metrics(worst_game_recall=0.70)
        second = _metrics(worst_game_recall=0.85)
        self.assertFalse(_update_early_stopping(state, cfg, first, 10, evaluation))
        self.assertEqual(state["best_step"], 10)
        self.assertFalse(_update_early_stopping(state, cfg, second, 20, evaluation))
        self.assertEqual(state["best_step"], 20)
        self.assertEqual(state["bad_evaluation_count"], 0)

    def test_tie_breaker_loss_is_a_plateau(self) -> None:
        state = _early_stopping_defaults()
        cfg, evaluation = _early_cfg(), _constrained_cfg()
        _update_early_stopping(
            state, cfg, _metrics(worst_game_recall=0.85), 10, evaluation
        )
        stop = _update_early_stopping(
            state, cfg, _metrics(worst_game_recall=0.70), 20, evaluation
        )
        self.assertTrue(stop)
        self.assertEqual(state["bad_evaluation_count"], 1)
        self.assertEqual(state["best_step"], 10)

    def test_negative_p999_tie_break_is_followed(self) -> None:
        state = _early_stopping_defaults()
        cfg, evaluation = _early_cfg(), _constrained_cfg()
        _update_early_stopping(state, cfg, _metrics(negative_p999=-1.0), 10, evaluation)
        # A lower negative-score p99.9 is better; must reset patience.
        self.assertFalse(
            _update_early_stopping(
                state, cfg, _metrics(negative_p999=-3.0), 20, evaluation
            )
        )
        self.assertEqual(state["best_step"], 20)

    def test_ineligible_evaluation_counts_toward_patience(self) -> None:
        state = _early_stopping_defaults()
        cfg = _early_cfg(patience_evaluations=2)
        evaluation = _constrained_cfg()
        _update_early_stopping(state, cfg, _metrics(global_recall=0.85), 10, evaluation)
        self.assertEqual(state["best_step"], 10)
        # Best raw recall in the fixture, but the FPR gate rejects it: not an
        # improvement, and it must consume patience rather than reset it.
        stop = _update_early_stopping(
            state, cfg, _metrics(global_fpr=0.5, global_recall=0.99), 20, evaluation
        )
        self.assertFalse(stop)
        self.assertEqual(state["bad_evaluation_count"], 1)
        self.assertEqual(state["best_step"], 10)
        stop = _update_early_stopping(
            state, cfg, _metrics(global_fpr=0.5, global_recall=0.99), 30, evaluation
        )
        self.assertTrue(stop)
        self.assertEqual(state["bad_evaluation_count"], 2)

    def test_first_evaluation_may_not_be_an_ineligible_best(self) -> None:
        state = _early_stopping_defaults()
        cfg, evaluation = _early_cfg(), _constrained_cfg()
        stop = _update_early_stopping(
            state, cfg, _metrics(global_fpr=0.5), 10, evaluation
        )
        self.assertTrue(stop)  # patience 1 exhausted immediately
        self.assertIsNone(state["best_value"])
        self.assertIsNone(state["best_step"])

    def test_min_delta_applies_to_the_primary_component_only(self) -> None:
        state = _early_stopping_defaults()
        cfg = _early_cfg(min_delta=0.01, patience_evaluations=5)
        evaluation = _constrained_cfg()
        _update_early_stopping(
            state,
            cfg,
            _metrics(global_recall=0.90, worst_game_recall=0.80),
            10,
            evaluation,
        )
        # +0.005 recall is inside the dead-band and the tie-breakers are flat.
        _update_early_stopping(
            state,
            cfg,
            _metrics(global_recall=0.905, worst_game_recall=0.80),
            20,
            evaluation,
        )
        self.assertEqual(state["bad_evaluation_count"], 1)
        # Still inside the dead-band, and the tie-breaker gain (+0.005) is
        # itself *smaller* than min_delta: min_delta must not mute the
        # tie-breakers, nor apply its dead-band to them.
        _update_early_stopping(
            state,
            cfg,
            _metrics(global_recall=0.905, worst_game_recall=0.805),
            30,
            evaluation,
        )
        self.assertEqual(state["bad_evaluation_count"], 0)
        self.assertEqual(state["best_step"], 30)
        # A primary gain beyond min_delta improves as well.
        _update_early_stopping(
            state,
            cfg,
            _metrics(global_recall=0.95, worst_game_recall=0.10),
            40,
            evaluation,
        )
        self.assertEqual(state["best_step"], 40)

    def test_burn_in_defers_the_stop(self) -> None:
        state = _early_stopping_defaults()
        cfg = _early_cfg(burn_in_steps=100)
        evaluation = _constrained_cfg()
        _update_early_stopping(
            state, cfg, _metrics(worst_game_recall=0.9), 10, evaluation
        )
        self.assertFalse(
            _update_early_stopping(
                state, cfg, _metrics(worst_game_recall=0.1), 20, evaluation
            )
        )
        self.assertEqual(state["bad_evaluation_count"], 1)

    def test_explicit_numeric_monitor_keeps_the_float_comparison(self) -> None:
        # monitor: cross_entropy / mode: min is a legitimate config and must
        # not be routed through the selection contract.
        state = _early_stopping_defaults()
        cfg = _early_cfg(monitor="cross_entropy", mode="min", patience_evaluations=2)
        evaluation = _constrained_cfg()
        _update_early_stopping(state, cfg, _metrics(cross_entropy=1.0), 10, evaluation)
        self.assertEqual(state["best_value"], 1.0)
        _update_early_stopping(state, cfg, _metrics(cross_entropy=0.8), 20, evaluation)
        self.assertEqual(state["best_value"], 0.8)
        self.assertEqual(state["best_step"], 20)
        _update_early_stopping(state, cfg, _metrics(cross_entropy=0.9), 30, evaluation)
        self.assertEqual(state["bad_evaluation_count"], 1)
        # An ineligible payload is irrelevant to a named numeric monitor.
        _update_early_stopping(
            state,
            cfg,
            _metrics(global_fpr=0.9, cross_entropy=0.5),
            40,
            evaluation,
        )
        self.assertEqual(state["best_value"], 0.5)
        self.assertEqual(state["bad_evaluation_count"], 0)

    def test_missing_monitor_metric_abstains(self) -> None:
        state = _early_stopping_defaults()
        cfg = _early_cfg(monitor="cross_entropy", mode="min")
        self.assertFalse(_update_early_stopping(state, cfg, {}, 10, _constrained_cfg()))
        self.assertEqual(state["bad_evaluation_count"], 0)
        self.assertIsNone(state["best_step"])

    def test_selection_path_abstains_instead_of_recording_a_verdict(self) -> None:
        # A skipped or non-rank-0 evaluation yields no payload. Abstention has
        # to leave the state untouched: burning patience here would stop a
        # healthy run, and writing the verdict sentinel into best_value would
        # poison every later comparison.
        state = _early_stopping_defaults()
        cfg, evaluation = _early_cfg(), _constrained_cfg()
        self.assertFalse(_update_early_stopping(state, cfg, {}, 10, evaluation))
        self.assertEqual(state, _early_stopping_defaults())
        # Non-empty payload that the selection config cannot rank either.
        self.assertFalse(
            _update_early_stopping(
                state,
                cfg,
                {"cross_entropy": 1.0},
                20,
                {"selection_metric": "a_metric_this_payload_lacks"},
            )
        )
        self.assertEqual(state, _early_stopping_defaults())

    def test_numeric_monitor_ignores_a_rank_key_left_by_another_monitor(self) -> None:
        # Resuming a run after switching monitor from the selection contract to
        # a scalar finds a rank-key list in best_value; it is not comparable to
        # a float, so the next evaluation becomes the new incumbent.
        state = _early_stopping_defaults()
        state.update({"best_value": [0.9, 0.8, 2.0], "best_step": 10})
        cfg = _early_cfg(monitor="cross_entropy", mode="min")
        self.assertFalse(
            _update_early_stopping(
                state, cfg, _metrics(cross_entropy=1.5), 20, _constrained_cfg()
            )
        )
        self.assertEqual(state["best_value"], 1.5)
        self.assertEqual(state["best_step"], 20)

    def test_metric_mode_matches_the_legacy_scalar_behavior(self) -> None:
        # In metric mode the rank key is the scalar selection metric, so the
        # decisions must be exactly the pre-step6 ones.
        state = _early_stopping_defaults()
        cfg = _early_cfg(patience_evaluations=2)
        evaluation = {"selection_metric": "global_f1_at_decision_threshold"}
        payloads = [0.5, 0.6, 0.55, 0.4]
        stops = [
            _update_early_stopping(
                state,
                cfg,
                {"global_f1_at_decision_threshold": value, "selection_score": value},
                (index + 1) * 10,
                evaluation,
            )
            for index, value in enumerate(payloads)
        ]
        self.assertEqual(stops, [False, False, False, True])
        self.assertEqual(state["best_step"], 20)
        self.assertEqual(state["best_value"], [0.6])

    def test_resume_from_a_scalar_best_value_keeps_patience(self) -> None:
        # Runs started before step6 persisted a float best_value; resume must
        # keep comparing against it rather than accept the next evaluation.
        state = _early_stopping_defaults()
        state.update({"best_value": 0.95, "best_step": 10})
        cfg, evaluation = _early_cfg(), _constrained_cfg()
        stop = _update_early_stopping(
            state, cfg, _metrics(global_recall=0.90), 20, evaluation
        )
        self.assertTrue(stop)
        self.assertEqual(state["best_step"], 10)
        self.assertEqual(state["bad_evaluation_count"], 1)

    def test_monitor_label_names_the_key_it_actually_used(self) -> None:
        self.assertEqual(
            _early_stopping_monitor_label(_early_cfg()), "selection_rank_key"
        )
        self.assertEqual(
            _early_stopping_monitor_label(_early_cfg(monitor="cross_entropy")),
            "cross_entropy",
        )


@unittest.skipIf(torch is None, "torch is not installed")
class EarlyStoppingTests(unittest.TestCase):
    def test_plateau_stops_before_budget_and_restores_best(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/recipes/example_debug.yaml")
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
            config["early_stopping"] = {
                "enabled": True,
                "monitor": "selection_score",
                "mode": "max",
                "full_validation_only": True,
                "burn_in_steps": 0,
                "patience_evaluations": 1,
                "min_delta": 0.0,
                "restore_best": True,
            }
            # Deterministic evaluations: first improves, second does not.
            fake_metrics = [
                {
                    "selection_score": 0.5,
                    "global_f1_tau099": 0.5,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.4,
                    "global_f1_tau099": 0.4,
                    "cross_entropy": 1.2,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 40,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 1)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)
            early = result["evaluation_state"]["early_stopping"]
            self.assertEqual(early["stop_reason"], "validation_plateau")
            self.assertEqual(early["stopped_at_step"], 40)
            self.assertEqual(early["best_step"], 20)
            self.assertEqual(early["bad_evaluation_count"], 1)
            # Stopped well before the 60-step budget.
            self.assertEqual(result["global_step"], 40)
            checkpoints = Path(directory) / "run" / "checkpoints"
            for name in (
                "model_best_selection.pth",
                "model_best_val_loss.pth",
                "model_best_worst_game.pth",
                "model_last.pth",
            ):
                self.assertTrue((checkpoints / name).is_file(), f"missing {name}")
            status = Path(directory) / "run" / "status.json"
            payload = __import__("json").loads(status.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("early_stopped"))
            self.assertEqual(payload.get("early_stopped_step"), 40)

    def test_constrained_mode_stops_on_a_tie_breaker_regression(self) -> None:
        """The loop must feed evaluation config into the early-stop decision.

        Both evaluations carry the same ``selection_score``/global recall, so
        the pre-step6 scalar monitor read them as "no improvement, no
        regression" from step 20 onward and only stopped by accident. Here
        step 40 regresses on worst-game recall and step 60 on the negative
        tail, which the rank key sees and the scalar cannot.
        """
        import json

        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/recipes/example_debug.yaml")
            config["experiment"]["output_dir"] = str(Path(directory) / "run")
            config["train"].update(
                {
                    "max_steps": 90,
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
                    **_constrained_cfg(),
                }
            )
            config["early_stopping"] = _early_cfg(
                full_validation_only=True,
                patience_evaluations=2,
                restore_best=True,
            )
            fake_metrics = [
                dict(_metrics(worst_game_recall=0.85), checkpoint_step=20),
                dict(_metrics(worst_game_recall=0.70), checkpoint_step=40),
                dict(
                    _metrics(worst_game_recall=0.70, negative_p999=-1.0),
                    checkpoint_step=60,
                ),
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 2)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            stdout = io.StringIO()
            with (
                mock.patch(
                    "game_cls.engine.training.loop._run_evaluation",
                    side_effect=fake_run_evaluation,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                result = run_training(config)
            early = result["evaluation_state"]["early_stopping"]
            self.assertEqual(early["stop_reason"], "validation_plateau")
            self.assertEqual(early["stopped_at_step"], 60)
            self.assertEqual(early["best_step"], 20)
            self.assertEqual(early["bad_evaluation_count"], 2)
            # Stopped before the 90-step budget on the tie-breakers alone.
            self.assertEqual(result["global_step"], 60)
            # The rank key, not a scalar, is what patience remembers.
            self.assertEqual(early["best_value"], [0.9, 0.85, 2.0])
            # The printed stop reason must name the key it actually used.
            console = stdout.getvalue()
            self.assertIn("[EARLY-STOP] validation plateau:", console)
            self.assertIn("monitor=selection_rank_key", console)
            self.assertNotIn("monitor=selection_score", console)
            summary = json.loads(
                (Path(directory) / "run" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["early_stopping"]["best_value"], [0.9, 0.85, 2.0])

    def test_restore_best_loads_selection_weights(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/recipes/example_debug.yaml")
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
            config["early_stopping"] = {
                "enabled": True,
                "monitor": "selection_score",
                "mode": "max",
                "full_validation_only": True,
                "burn_in_steps": 0,
                "patience_evaluations": 1,
                "min_delta": 0.0,
                "restore_best": True,
            }
            fake_metrics = [
                {
                    "selection_score": 0.9,
                    "global_f1_tau099": 0.9,
                    "cross_entropy": 1.0,
                    "worst_game_f1_tau099": 0.3,
                    "checkpoint_step": 20,
                },
                {
                    "selection_score": 0.8,
                    "global_f1_tau099": 0.8,
                    "cross_entropy": 1.2,
                    "worst_game_f1_tau099": 0.2,
                    "checkpoint_step": 40,
                },
            ]
            calls = {"index": 0}

            def fake_run_evaluation(**kwargs):
                payload = fake_metrics[min(calls["index"], 1)]
                calls["index"] += 1
                return mock.Mock(metrics=dict(payload))

            with mock.patch(
                "game_cls.engine.training.loop._run_evaluation",
                side_effect=fake_run_evaluation,
            ):
                result = run_training(config)
            summary = __import__("json").loads(
                (Path(directory) / "run" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["restored_best"])
            self.assertEqual(
                summary["early_stopping"]["stop_reason"],
                "validation_plateau",
            )
            self.assertEqual(result["global_step"], 40)


if __name__ == "__main__":
    unittest.main()
