from __future__ import annotations

from typing import Any

from ..contracts.trainable import StateSelection, TrainableSelection
from .base import TrainablePolicyBase, _group_spec


class ModelDeclaredTrainablePolicy(TrainablePolicyBase):
    """Lets the model declare its own parameter groups.

    The model must implement ``trainable_parameter_groups() -> list[dict]`` where
    each dict has a ``name`` and ``parameter_names`` key. This is the escape
    hatch for exotic architectures that can't be expressed by token or regex
    selection (USERPLAN §7.4).
    """

    policy_name = "model_declared"

    def select(self, model: Any) -> TrainableSelection:
        if not hasattr(model, "trainable_parameter_groups"):
            raise RuntimeError(
                "ModelDeclaredTrainablePolicy requires the model to implement "
                "trainable_parameter_groups()."
            )
        declared = model.trainable_parameter_groups()
        trainable_names: list[str] = []
        groups = []
        for group in declared:
            names = list(group["parameter_names"])
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name in set(names)
            trainable_names.extend(names)
            groups.append(
                _group_spec(
                    group["name"],
                    names,
                    lr_multiplier=float(group.get("learning_rate_multiplier", 1.0)),
                )
            )
        if not trainable_names:
            raise RuntimeError("ModelDeclaredTrainablePolicy matched no parameters.")
        all_names = list(dict.fromkeys(name for name, _ in model.named_parameters()))
        frozen_names = [name for name in all_names if name not in set(trainable_names)]
        return TrainableSelection(
            groups=tuple(groups),
            frozen_parameter_names=tuple(frozen_names),
            trainable_state=StateSelection(
                parameter_keys=tuple(trainable_names), buffer_keys=()
            ),
            frozen_state=StateSelection(
                parameter_keys=tuple(frozen_names), buffer_keys=()
            ),
        )

    def configure_module_modes(self, model: Any, selection: TrainableSelection) -> None:
        del selection
        model.eval()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                module = dict(model.named_modules()).get(name.rsplit(".", 1)[0] if "." in name else name)
                if module is not None:
                    module.train()

    def validate_loaded_state(
        self, model: Any, load_report: Any, selection: TrainableSelection
    ) -> float:
        del model, load_report, selection
        return 1.0
