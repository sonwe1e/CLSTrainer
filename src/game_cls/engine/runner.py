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
        1. Build the runtime (single instance).
        2. Call runtime.setup() (initializes device + process group).
        3. Seed the RNG (so model initialization is deterministic).
        4. Build components (model, task, policy, evaluator) using the SAME runtime.

        IMPORTANT: There must be exactly ONE runtime instance. It is created here,
        passed to build_core_components, used by the adapter for training, and
        cleaned up in close(). Previously two runtimes were created (one here, one
        inside build_core_components) and setup() was never called — that silently
        broke NPU device init and DDP/HCCL process group setup.
        """
        from game_cls.engine.trainer import _seed_everything

        # Build the single runtime instance.
        from .builders import build_runtime, _runtime_selector
        runtime = build_runtime(_runtime_selector(self._config))

        # Initialize the runtime: sets up the device (e.g. torch.npu.set_device)
        # and the distributed process group (e.g. dist.init_process_group).
        runtime.setup()

        rank = int(runtime.distributed.rank)

        # Seed BEFORE building the model so initialization is deterministic.
        seed = int(self._config["experiment"]["seed"])
        _seed_everything(seed + rank)

        # Now build components (model init will use the seeded RNG).
        # Pass the SAME runtime instance — do not let build_core_components
        # create a second one.
        self.components = build_core_components(self._config, runtime=runtime)
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
