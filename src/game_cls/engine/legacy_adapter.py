from __future__ import annotations

from typing import Any

from ..contracts.runtime import RuntimeStrategy


class LegacyTrainingEngineAdapter:
    """Adapts the legacy ``_run_training_loop`` to the runner's component model.

    The runner owns the runtime, task, trainable policy and evaluator. This
    adapter lets the runner delegate to the existing, battle-tested training
    loop without calling a private function directly. As the loop is
    progressively migrated to talk to components through their protocols
    (USERPLAN §9 R1–R3), this adapter shrinks until it can be removed.
    """

    def __init__(self, runtime: RuntimeStrategy) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> RuntimeStrategy:
        return self._runtime

    def train(
        self,
        config: dict[str, Any],
        *,
        task: Any = None,
        trainable_policy: Any = None,
        trainable_selection: Any = None,
        model: Any = None,
        evaluator: Any = None,
        image_spec: Any = None,
        data_module: Any = None,
    ) -> dict:
        """Run the legacy training loop using the adapter's runtime.

        The ``task``, ``trainable_policy``, ``trainable_selection``, ``model``,
        ``evaluator``, ``image_spec`` and ``data_module`` arguments are the
        components built by the runner. Passing them here means the loop must
        not rebuild them from scratch — the configured components are the
        single source of truth.
        """
        from .trainer import _run_training_loop

        return _run_training_loop(
            config,
            runtime=self._runtime,
            task=task,
            trainable_policy=trainable_policy,
            trainable_selection=trainable_selection,
            model=model,
            evaluator=evaluator,
            image_spec=image_spec,
            data_module=data_module,
        )
