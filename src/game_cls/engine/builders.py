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
    """Build a TaskAdapter from a task selector using the registry."""
    from ..registry import resolve

    task_type = str(selector.type)
    factory = resolve("task", task_type)
    return factory(selector, image_spec=image_spec, loss_config=loss_config)


def build_core_components(config: dict[str, Any]) -> ExperimentComponents:
    """Build the components that don't depend on dataloaders or resume state.

    This is the single place that turns a raw config into the extensible
    component graph (USERPLAN §9.3). It reuses the existing model/runtime
    construction behind the new interfaces.

    IMPORTANT: this function must be called AFTER the runtime is set up and
    the seed has been set. Model initialization is sensitive to the RNG
    state, and for exact training resume to work, every run must see the
    same initial weights given the same seed.
    """
    from game_cls.data.image_spec import ImageSpec

    from ..evaluation.legacy_adapter import LegacyEvaluatorSuite

    image_spec = ImageSpec.from_config(config["data"])
    runtime = build_runtime(_runtime_selector(config))

    # Build task via registry so that config task.type can select different
    # task implementations.
    task_selector = _task_selector(config)
    task = build_task(task_selector, image_spec=image_spec, loss_config=config["loss"])

    trainable_policy = build_trainable_policy(_trainable_selector(config))

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

    # Build the evaluator suite. During the migration we use the legacy
    # adapter which wraps the battle-tested ``evaluate`` function behind the
    # EvaluatorSuite interface (USERPLAN §10 E1).
    # Support both V2 (evaluation.decision.params.threshold) and legacy
    # (evaluation.threshold) config layouts.
    evaluation_cfg = config.get("evaluation", {})
    decision_cfg = evaluation_cfg.get("decision", {})
    decision_params = decision_cfg.get("params", {}) if isinstance(decision_cfg, dict) else {}
    threshold = float(
        decision_params.get("threshold")
        if isinstance(decision_params, dict) and decision_params.get("threshold") is not None
        else evaluation_cfg.get("threshold", 0.99)
    )
    evaluator = LegacyEvaluatorSuite(
        threshold=threshold,
        amp=bool(evaluation_cfg.get("amp", False)),
        amp_dtype=str(evaluation_cfg.get("amp_dtype", "bfloat16")),
        full_auc_mode=str(evaluation_cfg.get("full_auc_mode", "histogram")),
        auc_histogram_bins=int(evaluation_cfg.get("auc_histogram_bins", 4096)),
        quick_error_limit=int(evaluation_cfg.get("quick_save_error_limit", 200)),
        parquet_row_group_size=int(evaluation_cfg.get("parquet_row_group_size", 4096)),
    )

    return ExperimentComponents(
        config=_extract_structured_config(config),
        raw_config=config,
        runtime=runtime,
        task=task,
        trainable_policy=trainable_policy,
        trainable_selection=selection,
        model=model,
        image_spec=image_spec,
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
