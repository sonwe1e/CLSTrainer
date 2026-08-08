"""Staged partial unfreeze via per-rule training rules.

Uses a set of rules,
each matching parameters by regex and optionally deferring their unfreeze
until a global step, with a per-group learning-rate scale::

    model:
      trainable_rules:
        cls_head:
          pattern: "^cls\\."
          lr_scale: 1.0
          unfreeze_at_step: 0
        backbone_stage4:
          pattern: "^backbone\\.stage4\\."
          lr_scale: 0.10
          unfreeze_at_step: 1000

A parameter is trainable iff ``global_step >= unfreeze_at_step`` for the
highest-priority rule that matches it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainableRule:
    name: str
    pattern: str
    lr_scale: float = 1.0
    unfreeze_at_step: int = 0
    priority: int | None = None

    def matches(self, parameter_name: str) -> bool:
        return re.search(self.pattern, parameter_name) is not None


def parse_rules(rules_cfg: dict) -> list[TrainableRule]:
    """Build ordered rules (priority descending, then insertion order)."""
    parsed = []
    for name, cfg in (rules_cfg or {}).items():
        parsed.append(
            TrainableRule(
                name=str(name),
                pattern=str(cfg.get("pattern", "")),
                lr_scale=float(cfg.get("lr_scale", 1.0)),
                unfreeze_at_step=int(cfg.get("unfreeze_at_step", 0)),
                priority=(
                    int(cfg["priority"]) if cfg.get("priority") is not None else None
                ),
            )
        )
    parsed.sort(
        key=lambda rule: rule.priority if rule.priority is not None else -1,
        reverse=True,
    )
    return parsed


def rules_fingerprint(rules: list[TrainableRule]) -> str:
    """Stable identity of the rule set (resume/gate decisions depend on it)."""
    payload = [
        {
            "name": rule.name,
            "pattern": rule.pattern,
            "lr_scale": rule.lr_scale,
            "unfreeze_at_step": rule.unfreeze_at_step,
            "priority": rule.priority,
        }
        for rule in sorted(rules, key=lambda item: item.name)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def rule_for(rules: list[TrainableRule], parameter_name: str) -> TrainableRule | None:
    for rule in rules:
        if rule.matches(parameter_name):
            return rule
    return None


def resolve_trainable_names(
    model,
    rules: list[TrainableRule],
    step: int,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return (trainable_names, {rule_name: [names]}) at ``step``."""
    trainable_names: list[str] = []
    by_rule: dict[str, list[str]] = {}
    for name, _ in model.named_parameters():
        rule = rule_for(rules, name)
        if rule is None or step < rule.unfreeze_at_step:
            continue
        trainable_names.append(name)
        by_rule.setdefault(rule.name, []).append(name)
    return trainable_names, by_rule


def apply_trainable_state(model, rules: list[TrainableRule], step: int) -> None:
    """Set ``requires_grad`` for the trainable set at ``step``.

    Parameters not matched by any rule are always frozen.
    """
    matched_any = False
    for name, parameter in model.named_parameters():
        rule = rule_for(rules, name)
        trainable = rule is not None and step >= rule.unfreeze_at_step
        parameter.requires_grad = trainable
        matched_any = matched_any or (rule is not None)
    if not matched_any:
        raise RuntimeError(
            "No model parameter matches any trainable_rules pattern; the "
            "model would have no trainable parameters."
        )


def build_optimizer_parameter_groups(
    model,
    rules: list[TrainableRule],
    *,
    step: int,
    weight_decay: float,
    base_lr: float,
) -> list[dict]:
    """One optimizer group per matched rule, decay/no-decay split inside.

    Each group carries ``param_names`` (used for identity-safe optimizer
    restore on resume) and ``lr = base_lr * lr_scale``.
    """
    by_rule: dict[str, list[tuple[str, Any]]] = {}
    for name, parameter in model.named_parameters():
        rule = rule_for(rules, name)
        if rule is None or step < rule.unfreeze_at_step:
            continue
        by_rule.setdefault(rule.name, []).append((name, parameter))
    groups: list[dict] = []
    for rule in rules:
        if rule.name not in by_rule:
            continue
        decay: list[Any] = []
        no_decay: list[Any] = []
        names_decay: list[str] = []
        names_no_decay: list[str] = []
        for name, parameter in by_rule[rule.name]:
            if parameter.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(parameter)
                names_no_decay.append(name)
            else:
                decay.append(parameter)
                names_decay.append(name)
        lr = float(base_lr) * float(rule.lr_scale)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "param_names": names_decay,
                    "weight_decay": weight_decay,
                    "lr": lr,
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "param_names": names_no_decay,
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )
    if not groups:
        raise RuntimeError(
            "trainable_rules resolved to no optimizer groups at "
            f"step {step}; every rule is frozen at this step."
        )
    return groups
