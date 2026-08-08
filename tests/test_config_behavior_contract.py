"""Behavior-based config consumption (audit G1, SS7).

The leaf-string scan in ``test_config_consumption.py`` proves only that a
field's NAME appears somewhere in the runtime source; a consumer reading the
WRONG section (e.g. ``train.warmup_steps`` instead of
``scheduler.warmup_steps``) still passes, so it delivered false coverage
confidence. These tests mutate a field and assert the corresponding behavior
actually changes -- the contract the audit asks for ("modify this field, the
behavior must change").
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ConfigBehaviorContractTests(unittest.TestCase):
    def test_warmup_steps_moves_the_lr_schedule(self) -> None:
        # scheduler.warmup_steps must drive the LR ramp; a different warmup at
        # the same step must yield a different LR. The pre-fix dead read of
        # train.warmup_steps made this field inert.
        from game_cls.engine.training.state import _build_scheduler

        def lr_at(warmup: int, step: int) -> float:
            param = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.SGD([param], lr=1.0)
            scheduler = _build_scheduler(
                optimizer,
                {"warmup_steps": warmup, "min_learning_rate": 0.0},
                total_steps=100,
            )
            for _ in range(step):
                # optimizer.step() first avoids the "scheduler.step before
                # optimizer.step" warning; the LR is what we read back.
                optimizer.step()
                scheduler.step()
            return float(optimizer.param_groups[0]["lr"])

        self.assertNotEqual(lr_at(10, 5), lr_at(50, 5))
        # Warmup ramps monotonically: more warmup at the same early step means
        # a smaller multiplier.
        self.assertLess(lr_at(50, 5), lr_at(10, 5))

    def test_selection_mode_changes_the_rank_key(self) -> None:
        # selection_mode must change the best-checkpoint rank key: constrained
        # ranks by (recall, worst-recall, -negative p99.9), metric by the
        # scalar F1. The old production preset exposed dead composite weights
        # that could not influence the constrained key (audit P1-4).
        from game_cls.engine.training.selection import selection_sort_value

        metrics = {
            "global_positive_recall_at_decision_threshold": 0.9,
            "worst_game_positive_recall_at_decision_threshold": 0.8,
            "negative_score_p999": 0.5,
            "global_f1_at_decision_threshold": 0.7,
            "global_fpr_at_decision_threshold": 0.005,
            "worst_game_fpr_at_decision_threshold": 0.01,
        }
        constrained = selection_sort_value(
            metrics,
            {
                "selection_mode": "constrained",
                "max_global_fpr": 0.01,
                "max_worst_game_fpr": 0.02,
                "min_positive_recall": 0.8,
            },
        )
        metric = selection_sort_value(
            metrics,
            {
                "selection_mode": "metric",
                "selection_metric": "global_f1_at_decision_threshold",
            },
        )
        self.assertNotEqual(constrained, metric)

    def test_burn_in_steps_gates_patience_start(self) -> None:
        # early_stopping.burn_in_steps must gate when patience counting
        # begins: inside burn-in a bad evaluation is not counted, past it is
        # (audit P1-1). A config with a different burn-in behaves differently.
        from game_cls.engine.training.early_stopping import (
            _early_stopping_defaults,
            _update_early_stopping,
        )

        def bad_count(burn_in: int) -> int:
            state = _early_stopping_defaults()
            cfg = {
                "enabled": True,
                "monitor": "selection_score",
                "mode": "max",
                "burn_in_steps": burn_in,
                "patience_evaluations": 3,
                "min_delta": 0.0,
            }
            evaluation = {
                "selection_mode": "constrained",
                "max_global_fpr": 0.01,
                "max_worst_game_fpr": 0.02,
                "min_positive_recall": 0.8,
            }
            bad = {
                "global_fpr_at_decision_threshold": 0.5,
                "global_positive_recall_at_decision_threshold": 0.5,
                "worst_game_positive_recall_at_decision_threshold": 0.5,
                "negative_score_p999": 0.5,
                "selection_score": 0.5,
            }
            _update_early_stopping(state, cfg, bad, 10, evaluation)
            return state["bad_evaluation_count"]

        self.assertEqual(bad_count(burn_in=0), 1)
        self.assertEqual(bad_count(burn_in=1000), 0)

    def test_identity_mode_changes_the_source_uid(self) -> None:
        # data.source_video_identity.mode must change the audit identity: the
        # label-qualified mode yields a different stable_source_id than the
        # label-independent default (audit P0-5 / step7).
        from game_cls.data.splitter import stable_source_id

        plain = stable_source_id("MC", "01", 0, mode="game_video")
        label_aware = stable_source_id("MC", "01", 0, mode="game_label_video")
        self.assertNotEqual(plain, label_aware)
        self.assertTrue(label_aware.startswith("MC::0::01"))

    def test_periodic_state_mode_changes_checkpoint_contents(self) -> None:
        # checkpoint.periodic_state_mode must change what the checkpoint
        # carries: trainable_only persists a subset of the model, full
        # persists everything.
        from game_cls.engine.checkpoint import save_checkpoint_pair
        from game_cls.model.builder import build_demo_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = build_demo_model({})
            # The demo model's parameters default to all-trainable; freeze the
            # backbone so trainable_only and full actually differ.
            from game_cls.model.trainable_rules import (
                apply_trainable_state,
                parse_rules,
            )

            apply_trainable_state(
                model,
                parse_rules({"head": {"pattern": r"^cls\.", "lr_scale": 1.0}}),
                0,
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            full_keys: set[str] | None = None
            trainable_keys: set[str] | None = None
            for mode, marker in (("full", "f"), ("trainable_only", "t")):
                save_checkpoint_pair(
                    root,
                    marker,
                    model,
                    optimizer,
                    None,
                    None,
                    1,
                    10,
                    {},
                    {"model": {"checkpoint_path": None}},
                    state_mode=mode,
                )
                payload = torch.load(
                    root / f"checkpoint_{marker}.pth",
                    map_location="cpu",
                    weights_only=True,
                )
                keys = set(payload["model"])
                if mode == "full":
                    full_keys = keys
                else:
                    trainable_keys = keys
            assert full_keys is not None and trainable_keys is not None
            self.assertNotEqual(full_keys, trainable_keys)
            self.assertLess(len(trainable_keys), len(full_keys))


if __name__ == "__main__":
    unittest.main()
