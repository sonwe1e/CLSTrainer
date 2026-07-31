from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.evaluation import EvaluatorSuite
from ..contracts.runtime import RuntimeStrategy
from ..contracts.task import TaskAdapter
from ..contracts.trainable import TrainablePolicy, TrainableSelection
from ..runtime.factories import build_runtime
from ..trainable.build import build_trainable_policy


@dataclass
class ExperimentComponents:
    config: Any
    raw_config: dict[str, Any]
    runtime: RuntimeStrategy
    task: TaskAdapter
    trainable_policy: TrainablePolicy
    trainable_selection: TrainableSelection
    model: Any
    image_spec: Any
    data_module: Any = None
    evaluator: EvaluatorSuite | None = None
    load_report: Any = None
    # The following are built lazily by the runner because they depend on
    # dataloaders / resume state that is only known at run time.
    optimizer: Any = None
    scheduler: Any = None
    scaler: Any = None


def _build_model(config: dict[str, Any]) -> Any:
    from game_cls.model.builder import build_model

    model = build_model(config["model"])
    return model


def _load_base_checkpoint(model: Any, config: dict[str, Any]) -> Any:
    from game_cls.model.checkpoint_loader import load_model_checkpoint

    checkpoint_path = config["model"].get("checkpoint_path")
    report = None
    if checkpoint_path:
        report = load_model_checkpoint(model, checkpoint_path)
    return report


def build_task(selector: Any, *, image_spec: Any, loss_config: Any) -> Any:
    """Build a TaskAdapter from a task selector.

    Supports both registry-based resolution (``type``) and dynamic factory-path
    import (``factory``) so users can plug in custom tasks without modifying
    the registry (USERPLAN §12 factory-path support).
    """
    # Ensure all built-in tasks are registered.
    import game_cls.tasks.dual_frame_binary  # noqa: F401

    from ..registry import resolve_component

    factory = resolve_component(
        "task",
        str(selector.type),
        factory_path=getattr(selector, "factory", "") or "",
    )
    return factory(selector, image_spec=image_spec, loss_config=loss_config)


def _build_data_module(config: dict[str, Any], image_spec: Any) -> Any:
    """Build the DataModule from config.

    Uses the ``data.module_factory`` selector (if present) to dynamically
    resolve the DataModule implementation, falling back to the default
    :class:`LegacyGameVideoDataModule`. This allows the data pipeline to be
    swapped without modifying the builder.
    """
    from ..registry import import_from_path

    data_cfg = config.get("data", {})
    factory_path = str(data_cfg.get("module_factory", ""))
    if factory_path:
        try:
            factory = import_from_path(factory_path)
            return factory(config, image_spec)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Failed to import DataModule factory {factory_path!r}: {exc}"
            ) from exc

    from ..data.module import LegacyGameVideoDataModule

    return LegacyGameVideoDataModule(config, image_spec)


def build_core_components(
    config: dict[str, Any],
    *,
    runtime: Any = None,
) -> ExperimentComponents:
    """Build the components that don't depend on dataloaders or resume state.

    This is the single place that turns a raw config into the extensible
    component graph (USERPLAN §9.3). It reuses the existing model/runtime
    construction behind the new interfaces.

    IMPORTANT: this function must be called AFTER the runtime is set up and
    the seed has been set. Model initialization is sensitive to the RNG
    state, and for exact training resume to work, every run must see the
    same initial weights given the same seed.

    Args:
        config: The raw configuration dict.
        runtime: The already-set-up runtime instance. When provided (the normal
            runner path), this function reuses it instead of creating a new one.
            There must be exactly ONE runtime per training run.
    """
    from game_cls.data.image_spec import ImageSpec

    from ..evaluation.legacy_adapter import LegacyEvaluatorSuite

    image_spec = ImageSpec.from_config(config["data"])
    if runtime is None:
        # Standalone path: create a runtime here. The runner path always passes
        # in the runtime it already set up so there is only one instance.
        runtime = build_runtime(_runtime_selector(config))

    # Build task via registry so that config task.type can select different
    # task implementations.
    task_selector = _task_selector(config)
    task = build_task(task_selector, image_spec=image_spec, loss_config=config["loss"])

    trainable_policy = build_trainable_policy(_trainable_selector(config))

    # Build the data module. The default implementation wraps the legacy
    # data pipeline behind the DataModule interface.
    data_module = _build_data_module(config, image_spec)

    # Model is built AFTER seed is set (by the caller), so initialization
    # is deterministic and reproducible across runs.
    model = _build_model(config)

    # Load base checkpoint exactly once. The training loop must NOT load
    # it again.
    load_report = _load_base_checkpoint(model, config)

    selection = trainable_policy.select(model)
    trainable_policy.configure_module_modes(model, selection)
    if load_report is not None and not config["data"].get("synthetic", False):
        trainable_policy.validate_loaded_state(model, load_report, selection)

    # Build the evaluator suite. The default is the legacy adapter which wraps
    # the battle-tested ``evaluate`` function behind the EvaluatorSuite
    # interface (USERPLAN §10 E1). The suite is selected by
    # ``evaluation.suite.type`` via the registry so that custom suites can be
    # plugged in without modifying the builder.
    evaluation_cfg = config.get("evaluation", {})
    suite_selector = _suite_selector(config)
    from ..registry import resolve_component

    suite_factory = resolve_component(
        "evaluation_suite",
        str(suite_selector.type),
        factory_path=getattr(suite_selector, "factory", "") or "",
    )
    # Build the suite with the decision threshold and AMP settings.
    decision_cfg = evaluation_cfg.get("decision", {})
    decision_params = decision_cfg.get("params", {}) if isinstance(decision_cfg, dict) else {}
    threshold = float(
        decision_params.get("threshold")
        if isinstance(decision_params, dict) and decision_params.get("threshold") is not None
        else evaluation_cfg.get("threshold", 0.99)
    )
    evaluator = suite_factory(
        threshold=threshold,
        amp=bool(evaluation_cfg.get("amp", False)),
        amp_dtype=str(evaluation_cfg.get("amp_dtype", "bfloat16")),
        full_auc_mode=str(evaluation_cfg.get("full_auc_mode", "histogram")),
        auc_histogram_bins=int(evaluation_cfg.get("auc_histogram_bins", 4096)),
        quick_error_limit=int(evaluation_cfg.get("quick_save_error_limit", 200)),
        parquet_row_group_size=int(evaluation_cfg.get("parquet_row_group_size", 4096)),
    )
    # Fail fast if the evaluator suite is incompatible with the task.
    _validate_task_evaluator_compatibility(task, evaluator)

    return ExperimentComponents(
        config=_extract_structured_config(config),
        raw_config=config,
        runtime=runtime,
        task=task,
        trainable_policy=trainable_policy,
        trainable_selection=selection,
        model=model,
        image_spec=image_spec,
        data_module=data_module,
        evaluator=evaluator,
        load_report=load_report,
    )


def _runtime_selector(config: dict[str, Any]) -> Any:
    """Build a minimal namespace matching the ``runtime`` config section."""
    runtime_cfg = config.get("runtime", {})
    accelerator = runtime_cfg.get("accelerator", {}).get("type", config.get("device", {}).get("accelerator", "cpu"))
    distributed_cfg = runtime_cfg.get("distributed", {})
    distributed = distributed_cfg.get("type", "single_process")
    distributed_params = dict(distributed_cfg.get("params", {}) or {})

    class _Sel:
        pass

    acc_sel = _Sel()
    acc_sel.type = accelerator
    dist_sel = _Sel()
    dist_sel.type = distributed
    dist_sel.params = distributed_params
    runtime_sel = _Sel()
    runtime_sel.accelerator = acc_sel
    runtime_sel.distributed = dist_sel
    return runtime_sel


def _task_selector(config: dict[str, Any]) -> Any:
    task_cfg = config.get("task", {})

    class _Sel:
        pass

    sel = _Sel()
    sel.type = task_cfg.get("type", "dual_frame_binary")
    sel.factory = task_cfg.get("factory", "")
    sel.params = dict(task_cfg.get("params", {}) or {})
    return sel


def _trainable_selector(config: dict[str, Any]) -> Any:
    trainable_cfg = config.get("trainable", {})
    policy = trainable_cfg.get("policy", {})

    class _Sel:
        pass

    sel = _Sel()
    sel.type = policy.get("type", "name_token")
    sel.factory = policy.get("factory", "")
    sel.params = dict(policy.get("params", {}) or {})
    return sel


def _suite_selector(config: dict[str, Any]) -> Any:
    evaluation_cfg = config.get("evaluation", {})
    suite_cfg = evaluation_cfg.get("suite", {})

    class _Sel:
        pass

    sel = _Sel()
    sel.type = suite_cfg.get("type", "legacy_binary")
    sel.factory = suite_cfg.get("factory", "")
    sel.params = dict(suite_cfg.get("params", {}) or {})
    return sel


def _validate_task_evaluator_compatibility(task: Any, evaluator: Any) -> None:
    """Fail fast if the evaluator suite does not support the configured task."""
    supported = getattr(evaluator, "supported_task_names", None)
    if supported is None:
        return
    task_name = getattr(task, "task_name", None)
    if task_name is None:
        return
    if task_name not in supported:
        raise RuntimeError(
            f"EvaluatorSuite {type(evaluator).__name__!r} does not support "
            f"task {task_name!r}. Supported tasks: {sorted(supported)}"
        )


def _extract_structured_config(config: dict[str, Any]) -> Any:
    """Return a namespace view of the config sections the runner uses."""

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.experiment = config.get("experiment", {})
    cfg.device = config.get("device", {})
    cfg.train = config.get("train", {})
    cfg.evaluation = config.get("evaluation", {})
    cfg.checkpoint = config.get("checkpoint", {})
    cfg.optimizer = config.get("optimizer", {})
    cfg.scheduler = config.get("scheduler", {})
    cfg.loss = config.get("loss", {})
    cfg.model = config.get("model", {})
    cfg.data = config.get("data", {})
    cfg.dataloader = config.get("dataloader", {})
    cfg.augmentation = config.get("augmentation", {})
    cfg.sampler = config.get("sampler", {})
    cfg.distributed = config.get("distributed", {})
    return cfg
