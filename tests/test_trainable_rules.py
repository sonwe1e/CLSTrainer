"""Staged partial unfreeze via trainable_rules (step5 P4)."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from game_cls.model.builder import build_demo_model
from game_cls.model.trainable_rules import (
    apply_trainable_state,
    build_optimizer_parameter_groups,
    parse_rules,
    resolve_trainable_names,
    rule_for,
    rules_fingerprint,
)

RULES = {
    "cls_head": {"pattern": r"^cls\.", "lr_scale": 1.0, "unfreeze_at_step": 0},
    "backbone_late": {
        "pattern": r"^backbone\.0\.",
        "lr_scale": 0.1,
        "unfreeze_at_step": 100,
    },
}


@unittest.skipIf(torch is None, "torch is not installed")
class TrainableRulesTests(unittest.TestCase):
    def test_parse_and_match(self) -> None:
        rules = parse_rules(RULES)
        self.assertIsNotNone(rule_for(rules, "cls.weight"))
        self.assertIsNotNone(rule_for(rules, "backbone.0.weight"))
        self.assertIsNone(rule_for(rules, "backbone.2.bias"))

    def test_apply_state_step0_frozen_until_unfreeze(self) -> None:
        rules = parse_rules(RULES)
        model = build_demo_model({})
        apply_trainable_state(model, rules, 0)
        by_name = dict(model.named_parameters())
        self.assertTrue(by_name["cls.weight"].requires_grad)
        self.assertFalse(by_name["backbone.0.weight"].requires_grad)

    def test_apply_state_after_unfreeze(self) -> None:
        rules = parse_rules(RULES)
        model = build_demo_model({})
        apply_trainable_state(model, rules, 200)
        by_name = dict(model.named_parameters())
        self.assertTrue(by_name["cls.weight"].requires_grad)
        self.assertTrue(by_name["backbone.0.weight"].requires_grad)

    def test_resolve_trainable_names_by_rule(self) -> None:
        rules = parse_rules(RULES)
        model = build_demo_model({})
        names, by_rule = resolve_trainable_names(model, rules, 0)
        self.assertIn("cls.weight", names)
        self.assertNotIn("backbone.0.weight", names)
        self.assertIn("cls.weight", by_rule["cls_head"])

    def test_optimizer_groups_lr_scale(self) -> None:
        rules = parse_rules(RULES)
        model = build_demo_model({})
        apply_trainable_state(model, rules, 200)
        groups = build_optimizer_parameter_groups(
            model, rules, step=200, weight_decay=1e-4, base_lr=0.001
        )
        lrs = {round(float(group["lr"]), 6) for group in groups}
        self.assertIn(0.001, lrs)  # cls_head scale 1.0
        self.assertIn(0.0001, lrs)  # backbone_late scale 0.1
        for group in groups:
            self.assertTrue(group["param_names"])

    def test_fingerprint_changes_with_rules(self) -> None:
        rules_a = parse_rules(RULES)
        rules_b = parse_rules({"cls_head": {"pattern": r"^cls\.", "lr_scale": 2.0}})
        self.assertNotEqual(rules_fingerprint(rules_a), rules_fingerprint(rules_b))
        self.assertEqual(
            rules_fingerprint(rules_a), rules_fingerprint(parse_rules(RULES))
        )

    def test_no_trainable_params_raises(self) -> None:
        model = build_demo_model({})
        with self.assertRaises(RuntimeError):
            apply_trainable_state(model, parse_rules({"none": {"pattern": r"^zzz"}}), 0)


if __name__ == "__main__":
    unittest.main()
