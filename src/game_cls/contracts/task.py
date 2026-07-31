from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class StepContext:
    global_step: int
    total_steps: int
    epoch: int
    device: Any
    use_amp: bool
    amp_dtype: str


@dataclass
class TaskOutput:
    raw: Any
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutput:
    total: Any
    components: Mapping[str, Any]

    def __iter__(self):
        return iter((self.total, self.components))


@dataclass
class PredictionBatch:
    scores: Any
    predictions: Any
    targets: Any
    metadata: Any
    extras: dict[str, Any] = field(default_factory=dict)


class TaskAdapter(Protocol):
    """The per-task contract: batch validation, device transfer, forward, loss, predictions.

    A TaskAdapter owns everything about the learning task that is not data, model
    architecture, or runtime. The default implementation is dual-frame binary
    classification, but a future task only needs to supply a new adapter.
    """

    @property
    def task_name(self) -> str: ...

    @property
    def contract_version(self) -> int: ...

    def validate_model(self, model: Any) -> None: ...

    def validate_cpu_batch(self, batch: Any) -> None: ...

    def move_batch_to_device(self, batch: Any, context: StepContext) -> Any: ...

    def forward(self, model: Any, device_batch: Any, context: StepContext) -> TaskOutput: ...

    def compute_loss(
        self, output: TaskOutput, device_batch: Any, context: StepContext
    ) -> LossOutput: ...

    def build_predictions(
        self, output: TaskOutput, device_batch: Any, context: StepContext
    ) -> PredictionBatch: ...
