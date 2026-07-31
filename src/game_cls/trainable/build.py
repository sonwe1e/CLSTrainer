from __future__ import annotations

from typing import Any

from ..contracts.trainable import TrainablePolicy
from ..registry import resolve_component
from .name_token import NameTokenTrainablePolicy
from .regex import RegexTrainablePolicy


def build_trainable_policy(selector: Any) -> TrainablePolicy:
    """Build a :class:`TrainablePolicy` from a ``trainable.policy`` selector.

    Supports both registry-based resolution (``type``) and dynamic factory-path
    import (``factory``) so users can plug in custom policies without modifying
    the registry.
    """
    # If a factory_path is provided, use it directly.
    factory_path = getattr(selector, "factory", "") or ""
    if factory_path:
        from ..registry import import_from_path

        return import_from_path(factory_path)(**(dict(selector.params) or {}))

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
            freeze_trainable_batchnorm_stats=bool(
                params.get("freeze_trainable_batchnorm_stats", True)
            ),
            freeze_frozen_batchnorm_stats=bool(
                params.get("freeze_frozen_batchnorm_stats", True)
            ),
        )
    if policy_type == "model_declared":
        from .model_declared import ModelDeclaredTrainablePolicy

        return ModelDeclaredTrainablePolicy()
    raise ValueError(f"Unknown trainable policy type: {policy_type}")
