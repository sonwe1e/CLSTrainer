from __future__ import annotations

from typing import Any

from . import trainer as _trainer
from .builders import ExperimentComponents, build_core_components
from .legacy_adapter import LegacyTrainingEngineAdapter
from .state import ExperimentState


class ExperimentRunner:
    """Orchestrates a training run from components (USERPLAN §9).

    The runner owns the training loop, evaluation and checkpointing, talking to
    the task, runtime and trainable policy only through their protocols. The
    default wiring reproduces the legacy ``run_training`` behavior exactly; a
    future task swaps components without touching this loop.
    """

    def __init__(self, components: ExperimentComponents) -> None:
        self.components = components
        self.state = ExperimentState()
        self._runtime = components.runtime
        self._task = components.task
        self._adapter = LegacyTrainingEngineAdapter(self._runtime)

    # -- public API --------------------------------------------------------
    def setup(self) -> None:
        self._runtime.setup()

    def run(self) -> dict:
        # Delegate to the legacy loop through the adapter, passing the
        # component runtime *and* the components the runner built. The loop
        # must use these components instead of rebuilding them from scratch —
        # this is how the configured task, trainable policy, model and
        # evaluator actually drive training (USERPLAN §9 R1–R3).
        return self._adapter.train(
            self.components.raw_config,
            task=self.components.task,
            trainable_policy=self.components.trainable_policy,
            model=self.components.model,
            evaluator=self.components.evaluator,
            image_spec=self.components.image_spec,
        )

    def run_train_step(self, batch: Any) -> Any:
        raise NotImplementedError("Per-step API reserved for future streaming runners.")

    def run_evaluation(self, kind: str) -> Any:
        raise NotImplementedError("Direct evaluation API reserved for future use.")

    def save_checkpoint(self, tag: str) -> None:
        raise NotImplementedError("Direct checkpoint API reserved for future use.")

    def close(self) -> None:
        self._runtime.cleanup()


def build_and_run(config: dict[str, Any]) -> dict:
    """Convenience: build components and run (used by the compat wrapper)."""
    components = build_core_components(config)
    runner = ExperimentRunner(components)
    try:
        runner.setup()
        return runner.run()
    finally:
        runner.close()
