from __future__ import annotations

from typing import Any

from ..contracts.trainable import TrainablePolicy
from .name_token import NameTokenTrainablePolicy
from .regex import RegexTrainablePolicy


def build_trainable_policy(selector: Any) -> TrainablePolicy:
    """Build a :class:`TrainablePolicy` from a ``trainable.policy`` selector."""
    policy_type = str(selector.type)
    params = dict(selector.params) if selector.params else {}
    if policy_type == "name_token":
        return NameTokenTrainablePolicy(
            token=str(params.get("token", "cls")),
            case_sensitive=bool(params.get("case_sensitive", True)),
            freeze_trainable_batchnorm_stats=bool(
                params.get("freeze_trainable_batchnorm_stats", True)
            ),
            freeze_frozen_batchnorm_stats=bool(
                params.get("freeze_frozen_batchnorm_stats", True)
            ),
        )
    if policy_type == "regex":
        return RegexTrainablePolicy(
            include=list(params.get("include", [])),
            exclude=list(params.get("exclude", [])),
        )
    if policy_type == "model_declared":
        from .model_declared import ModelDeclaredTrainablePolicy

        return ModelDeclaredTrainablePolicy()
    raise ValueError(f"Unknown trainable policy type: {policy_type}")
