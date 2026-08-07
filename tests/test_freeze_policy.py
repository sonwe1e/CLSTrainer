from __future__ import annotations

import unittest

from game_cls.model.freeze_policy import (
    configure_trainable_parameters,
    set_frozen_backbone_train_mode,
)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.count = count
        self.requires_grad = True

    def numel(self) -> int:
        return self.count


class FakeModel:
    def __init__(self) -> None:
        self.parameters = {
            "backbone.weight": FakeParameter(100),
            "head.Classifier.weight": FakeParameter(10),
            "head.cls.weight": FakeParameter(8),
            "head.cls.bias": FakeParameter(2),
        }

    def named_parameters(self):
        return self.parameters.items()


class FreezePolicyTests(unittest.TestCase):
    def test_only_lowercase_cls_is_trainable(self) -> None:
        model = FakeModel()
        summary = configure_trainable_parameters(model)
        self.assertEqual(summary.trainable_names, ("head.cls.weight", "head.cls.bias"))
        self.assertEqual(summary.trainable_count, 10)
        self.assertEqual(summary.frozen_count, 110)
        self.assertFalse(model.parameters["head.Classifier.weight"].requires_grad)

    def test_fails_when_no_parameter_matches(self) -> None:
        model = FakeModel()
        with self.assertRaises(RuntimeError):
            configure_trainable_parameters(model, "missing")


@unittest.skipIf(torch is None, "torch is not installed")
class StagedUnfreezeTrainModeTests(unittest.TestCase):
    """Audit P1-2: an unfrozen stage must run in train mode.

    The legacy train-mode rule was ``"cls" in module_name``, so a staged
    unfreeze that opens ``stage4`` (whose names carry no "cls" token) left the
    freshly-trainable modules in eval mode: Dropout/DropPath never applied and
    the weights trained under inference semantics. Train mode must follow the
    ``requires_grad`` set (which reflects the active rules), not the name.
    """

    def _build(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(4, 4, 1),
            nn.BatchNorm2d(4),
            nn.Dropout(0.5),
        )

    def _model(self) -> nn.Module:
        return nn.ModuleDict(
            {
                "backbone": nn.ModuleDict(
                    {"stage3": self._build(), "stage4": self._build()}
                ),
                "cls_head": nn.Sequential(nn.Linear(4, 2), nn.Dropout(0.5)),
            }
        )

    def _rules(self):
        from game_cls.model.trainable_rules import parse_rules

        return parse_rules(
            {
                "cls_head": {
                    "pattern": "cls_head",
                    "lr_scale": 1.0,
                    "unfreeze_at_step": 0,
                    "priority": 0,
                },
                "backbone_late": {
                    "pattern": "stage4",
                    "lr_scale": 0.1,
                    "unfreeze_at_step": 4000,
                    "priority": 1,
                },
            }
        )

    def _apply(self, model, rules, step: int) -> None:
        from game_cls.model.trainable_rules import apply_trainable_state

        apply_trainable_state(model, rules, step)
        set_frozen_backbone_train_mode(
            model,
            "cls",
            None,
            freeze_backbone_batchnorm_stats=True,
            freeze_cls_batchnorm_stats=True,
        )

    def test_unfrozen_stage_trains_its_dropout(self) -> None:
        model = self._model()
        rules = self._rules()
        # Step 0: only the head trains.
        self._apply(model, rules, 0)
        self.assertTrue(model["cls_head"].training)
        self.assertFalse(model["backbone"]["stage3"].training)
        self.assertFalse(model["backbone"]["stage4"].training)
        # After the boundary: stage4 (and its Dropout) must enter train mode
        # even though its names carry no "cls" token.
        self._apply(model, rules, 4000)
        self.assertTrue(model["backbone"]["stage4"].training)
        self.assertTrue(model["backbone"]["stage4"][2].training)  # Dropout
        # The still-frozen stage3 stays in eval, Dropout deterministic.
        self.assertFalse(model["backbone"]["stage3"].training)
        self.assertFalse(model["backbone"]["stage3"][2].training)

    def test_frozen_stage_batchnorm_stats_stay_frozen(self) -> None:
        model = self._model()
        rules = self._rules()
        self._apply(model, rules, 4000)
        # stage4 BN is re-eval'd by the freeze_backbone_batchnorm_stats control.
        self.assertFalse(model["backbone"]["stage4"][1].training)

    def test_batchnorm_stats_control_independent_of_stage_train_mode(self) -> None:
        from game_cls.model.trainable_rules import apply_trainable_state

        model = self._model()
        rules = self._rules()
        apply_trainable_state(model, rules, 4000)
        set_frozen_backbone_train_mode(
            model,
            "cls",
            None,
            freeze_backbone_batchnorm_stats=False,
            freeze_cls_batchnorm_stats=True,
        )
        # The stage trains, its conv is train mode; BN stats may also update.
        self.assertTrue(model["backbone"]["stage4"].training)
        self.assertTrue(model["backbone"]["stage4"][1].training)


if __name__ == "__main__":
    unittest.main()
