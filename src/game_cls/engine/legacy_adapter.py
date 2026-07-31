from __future__ import annotations

from typing import Any

from ..contracts.runtime import RuntimeStrategy


class LegacyTrainingEngineAdapter:
    """Adapts the legacy ``_run_training_loop`` to the runner's component model.

    The runner owns the runtime, task and trainable policy. This adapter lets
    the runner delegate to the existing, battle-tested training loop without
    calling a private function directly. As the loop is progressively migrated
    to talk to components through their protocols (USERPLAN §9 R1–R3), this
    adapter shrinks until it can be removed.
    """

    def __init__(self, runtime: RuntimeStrategy) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> RuntimeStrategy:
        return self._runtime

    def train(self, config: dict[str, Any]) -> dict:
        """Run the legacy training loop using the adapter's runtime."""
        from .trainer import _run_training_loop

        return _run_training_loop(config, runtime=self._runtime)
