from __future__ import annotations

from game_cls.model.freeze_policy import (
    set_frozen_backbone_train_mode,
)


def _set_train_mode(model, model_config: dict) -> None:
    set_frozen_backbone_train_mode(
        model,
        "cls",
        freeze_backbone_batchnorm_stats=model_config.get(
            "freeze_backbone_batchnorm_stats", True
        ),
        freeze_cls_batchnorm_stats=model_config.get("freeze_cls_batchnorm_stats", True),
    )


def build_optimizer_parameter_groups(model, weight_decay: float) -> list[dict]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
