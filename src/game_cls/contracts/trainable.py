from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ParameterGroupSpec:
    name: str
    parameter_names: tuple[str, ...]
    learning_rate_multiplier: float = 1.0
    weight_decay: float | None = None


@dataclass(frozen=True)
class StateSelection:
    parameter_keys: tuple[str, ...]
    buffer_keys: tuple[str, ...]


@dataclass(frozen=True)
class TrainableSelection:
    groups: tuple[ParameterGroupSpec, ...]
    frozen_parameter_names: tuple[str, ...]
    trainable_state: StateSelection
    frozen_state: StateSelection


class TrainablePolicy(Protocol):
    @property
    def policy_name(self) -> str: ...

    @property
    def state_version(self) -> int: ...

    def select(self, model: Any) -> TrainableSelection: ...

    def configure_module_modes(self, model: Any, selection: TrainableSelection) -> None: ...

    def validate_loaded_state(
        self, model: Any, load_report: Any, selection: TrainableSelection
    ) -> float: ...
