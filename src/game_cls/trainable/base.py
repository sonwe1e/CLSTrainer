from __future__ import annotations

from typing import Any

from ..contracts.trainable import (
    ParameterGroupSpec,
    StateSelection,
    TrainablePolicy,
    TrainableSelection,
)


class TrainablePolicyBase:
    """Convenience base exposing the protocol's required attributes."""

    policy_name = "base"

    def select(self, model: Any) -> TrainableSelection:
        raise NotImplementedError

    def configure_module_modes(self, model: Any, selection: TrainableSelection) -> None:
        del model, selection

    def validate_loaded_state(
        self, model: Any, load_report: Any, selection: TrainableSelection
    ) -> float:
        del model, load_report, selection
        return 1.0


def _empty_selection() -> TrainableSelection:
    return TrainableSelection(
        groups=(),
        frozen_parameter_names=(),
        trainable_state=StateSelection(parameter_keys=(), buffer_keys=()),
        frozen_state=StateSelection(parameter_keys=(), buffer_keys=()),
    )


def _group_spec(name: str, parameter_names: list[str], lr_multiplier: float = 1.0) -> ParameterGroupSpec:
    return ParameterGroupSpec(
        name=name,
        parameter_names=tuple(parameter_names),
        learning_rate_multiplier=lr_multiplier,
    )
