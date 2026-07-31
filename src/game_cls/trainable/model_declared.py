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
        if not declared:
            raise RuntimeError("ModelDeclaredTrainablePolicy: no groups declared.")

        # Validate group names are unique.
        group_names = [g.get("name", "") for g in declared]
        duplicate_names = {n for n in group_names if group_names.count(n) > 1}
        if duplicate_names:
            raise RuntimeError(
                f"ModelDeclaredTrainablePolicy: duplicate group names: {duplicate_names}"
            )

        # Validate parameters exist and don't appear in multiple groups.
        parameters = dict(model.named_parameters())
        seen: set[str] = set()
        trainable_set: set[str] = set()
        groups = []
        for group in declared:
            names = list(group["parameter_names"])
            missing = set(names) - parameters.keys()
            if missing:
                raise RuntimeError(
                    f"ModelDeclaredTrainablePolicy: group {group.get('name')!r} "
                    f"contains missing parameters: {missing}"
                )
            overlap = seen.intersection(names)
            if overlap:
                raise RuntimeError(
                    f"ModelDeclaredTrainablePolicy: parameters appear in multiple "
                    f"groups: {overlap}"
                )
            trainable_set.update(names)
            seen.update(names)
            groups.append(
                _group_spec(
                    group["name"],
                    names,
                    lr_multiplier=float(group.get("learning_rate_multiplier", 1.0)),
                )
            )

        # First collect every trainable name across all groups, then set
        # requires_grad once. The previous loop set it per-group, which meant
        # each group froze the previous group's parameters (last-group-wins).
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in trainable_set
        if not trainable_set:
            raise RuntimeError("ModelDeclaredTrainablePolicy matched no parameters.")
        all_names = list(dict.fromkeys(name for name, _ in model.named_parameters()))
        frozen_names = [name for name in all_names if name not in trainable_set]
        return TrainableSelection(
            groups=tuple(groups),
            frozen_parameter_names=tuple(frozen_names),
            trainable_state=StateSelection(
                parameter_keys=tuple(sorted(trainable_set)), buffer_keys=()
            ),
            frozen_state=StateSelection(
                parameter_keys=tuple(frozen_names), buffer_keys=()
            ),
        )

    def configure_module_modes(self, model: Any, selection: TrainableSelection) -> None:
        # Derive module names from selected parameters.
        trainable_param_names = set(selection.trainable_state.parameter_keys)
        trainable_module_names = {
            name.rsplit(".", 1)[0] if "." in name else name
            for name in trainable_param_names
        }
        model.eval()
        for module_name, module in model.named_modules():
            if module_name in trainable_module_names:
                module.train()

    def validate_loaded_state(
        self, model: Any, load_report: Any, selection: TrainableSelection
    ) -> float:
        # Validate that the frozen backbone parameters are fully covered by the
        # loaded checkpoint. The frozen set is explicit in the selection.
        frozen_keys = set(selection.frozen_state.parameter_keys)
        if not frozen_keys:
            return 1.0
        loaded = set(getattr(load_report, "loaded", ()))
        missing = frozen_keys - loaded
        if missing:
            raise RuntimeError(
                "Production checkpoint must load 100% of the frozen backbone. "
                f"Missing {len(missing)} frozen parameter(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return 1.0
