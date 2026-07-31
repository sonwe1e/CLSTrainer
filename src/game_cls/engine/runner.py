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

    Lifecycle::

        runner = ExperimentRunner(config)
        runner.setup()   # runtime setup + seed + build components
        runner.run()     # training loop
        runner.close()   # cleanup
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self.components: ExperimentComponents | None = None
        self.state = ExperimentState()
        self._adapter: LegacyTrainingEngineAdapter | None = None

    # -- public API --------------------------------------------------------
    def setup(self) -> None:
        """Set up runtime, seed RNG, then build components.

        The order matters for exact training resume:
        1. Set up the runtime (so we know the rank).
        2. Seed the RNG (so model initialization is deterministic).
        3. Build components (model, task, policy, evaluator).
        """
        from game_cls.engine.trainer import _seed_everything

        # Build runtime first so we know the rank for seeding.
        from .builders import build_runtime, _runtime_selector
        runtime = build_runtime(_runtime_selector(self._config))
        rank = int(runtime.distributed.rank)

        # Seed BEFORE building the model so initialization is deterministic.
        seed = int(self._config["experiment"]["seed"])
        _seed_everything(seed + rank)

        # Now build components (model init will use the seeded RNG).
        self.components = build_core_components(self._config)
        self._adapter = LegacyTrainingEngineAdapter(runtime)

    def run(self) -> dict:
        # Delegate to the legacy loop through the adapter, passing the
        # component runtime *and* the components the runner built. The loop
        # must use these components instead of rebuilding them from scratch —
        # this is how the configured task, trainable policy, model and
        # evaluator actually drive training (USERPLAN §9 R1–R3).
        if self.components is None or self._adapter is None:
            raise RuntimeError("ExperimentRunner.setup() must be called before run().")
        return self._adapter.train(
            self.components.raw_config,
            task=self.components.task,
            trainable_policy=self.components.trainable_policy,
            trainable_selection=self.components.trainable_selection,
            model=self.components.model,
            evaluator=self.components.evaluator,
            image_spec=self.components.image_spec,
            data_module=self.components.data_module,
        )

    def run_train_step(self, batch: Any) -> Any:
        raise NotImplementedError("Per-step API reserved for future streaming runners.")

    def run_evaluation(self, kind: str) -> Any:
        raise NotImplementedError("Direct evaluation API reserved for future use.")

    def save_checkpoint(self, tag: str) -> None:
        raise NotImplementedError("Direct checkpoint API reserved for future use.")

    def close(self) -> None:
        if self.components is not None:
            self.components.runtime.cleanup()


def build_and_run(config: dict[str, Any]) -> dict:
    """Convenience: build components and run (used by the compat wrapper)."""
    runner = ExperimentRunner(config)
    try:
        runner.setup()
        return runner.run()
    finally:
        runner.close()
