from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError


class ValidationError(Exception):
    """Raised when a configuration fails schema validation or migration."""


def _wrap_pydantic_error(exc: PydanticValidationError) -> ValidationError:
    messages = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error["loc"])
        messages.append(f"{loc}: {error['msg']}")
    return ValidationError(
        "Configuration validation failed:\n  " + "\n  ".join(messages)
    )


def _field_names(model: type[BaseModel]) -> set[str]:
    return set(model.model_fields)


def _unknown_keys(data: Any, model: type[BaseModel], path: str = "") -> list[str]:
    """Return dotted paths for keys not present in the Pydantic model.

    Used to surface unknown top- and first-level plugin-param keys while still
    allowing arbitrary leaf params inside an explicit ``params`` dict.
    """
    unknown: list[str] = []
    if not isinstance(data, dict) or not issubclass(model, BaseModel):
        return unknown
    known = _field_names(model)
    for key, value in data.items():
        dotted = f"{path}.{key}" if path else key
        if key not in known:
            unknown.append(dotted)
            continue
        field_info = model.model_fields[key]
        target = field_info.annotation
        # Only descend one model level; plugin params stay open.
        if (
            isinstance(target, type)
            and issubclass(target, BaseModel)
            and isinstance(value, dict)
        ):
            unknown.extend(_unknown_keys(value, target, dotted))
    return unknown


# ----------------------------------------------------------------------- #
# Plugin parameter models (strict validation for each component's params)
# ----------------------------------------------------------------------- #
class _ThresholdDecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(ge=0.0, le=1.0)


class _NameTokenPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = "cls"
    case_sensitive: bool = True
    freeze_trainable_batchnorm_stats: bool = True
    freeze_frozen_batchnorm_stats: bool = True


class _RegexPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    freeze_trainable_batchnorm_stats: bool = True
    freeze_frozen_batchnorm_stats: bool = True


class _ModelDeclaredPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Model-declared policy takes no params; the model itself declares groups.


class _DdpRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = "nccl"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False
    gradient_as_bucket_view: bool = True


class _PngBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # PNG backend takes no params.


class _PackedBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index_path: str = ""
    max_open_shards: int = Field(default=16, ge=1)


class _BalancedSamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    class_probability: dict[int, float] = Field(default_factory=lambda: {0: 0.5, 1: 0.5})
    deduplicate_within_global_batch: bool = True


class _DualFrameBinaryTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive_class_index: int = 1
    num_classes: int = 2


# Mapping from (selector_type, plugin_type) → params Pydantic model.
# Used by ``_check_plugin_params`` to validate plugin params strictly so
# that a typo like ``threshhold`` fails fast instead of being silently
# ignored (USERPLAN §8.4).
_PLUGIN_PARAMS_MODELS: dict[tuple[str, str], type[BaseModel]] = {
    ("evaluation", "threshold"): _ThresholdDecisionConfig,
    ("trainable", "name_token"): _NameTokenPolicyConfig,
    ("trainable", "regex"): _RegexPolicyConfig,
    ("trainable", "model_declared"): _ModelDeclaredPolicyConfig,
    ("runtime", "ddp"): _DdpRuntimeConfig,
    ("data", "png"): _PngBackendConfig,
    ("data", "packed_uint8"): _PackedBackendConfig,
    ("sampler", "balanced_game_label_delta"): _BalancedSamplerConfig,
    ("task", "dual_frame_binary"): _DualFrameBinaryTaskConfig,
}


# ----------------------------------------------------------------------- #
# Plugin selector models
# ----------------------------------------------------------------------- #
class _PluginSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class _TaskSelector(_PluginSelector):
    factory: str = ""


class _TrainableSelector(_PluginSelector):
    factory: str = ""


class _IndexCodecSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class _BackendSelector(_PluginSelector):
    pass


class _SamplerSelector(_PluginSelector):
    pass


class _AcceleratorSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class _DistributedSelector(_PluginSelector):
    pass


class _DecisionSelector(_PluginSelector):
    pass


class _SuiteSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


# ----------------------------------------------------------------------- #
# Top-level V2 sections. The layout follows USERPLAN §8.4: selectors are
# nested one level (trainable.policy, sampler.policy, runtime.{accelerator,
# distributed}, evaluation.{suite, decision}, data.{index_codec, backend}).
# ----------------------------------------------------------------------- #
# Intermediate sections tolerate legacy flat keys (extra="allow") during the
# migration period. Strictness is enforced at the leaf component-selector
# level by _check_selector_keys, so a typo in a new plugin selector still
# fails fast while migrated V1 keys pass through.
class _TrainableConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    policy: _TrainableSelector


class _SamplerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    policy: _SamplerSelector


class _RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    accelerator: _AcceleratorSelector
    distributed: _DistributedSelector


class _EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite: _SuiteSelector
    decision: _DecisionSelector
    # Legacy flat evaluation keys (threshold, amp, selection_metric, ...).
    # Included here so that typos like ``threshhold`` fail fast.
    threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    amp: bool = False
    amp_dtype: str = "bfloat16"
    full_auc_mode: str = "histogram"
    auc_histogram_bins: int = Field(default=4096, ge=1)
    quick_save_error_limit: int = Field(default=200, ge=0)
    quick_test_pairs_per_video: int = Field(default=128, ge=1)
    parquet_row_group_size: int = Field(default=4096, ge=1)
    selection_metric: str = "global_f1_tau099"
    minimum_worst_game_f1: float | None = None
    selection_weights: dict[str, float] = Field(default_factory=dict)
    quick_test_every_steps: int = Field(default=0, ge=0)
    full_test_every_steps: int = Field(default=0, ge=0)
    full_test_at_end: bool = True
    html_max_errors_per_group: int = Field(default=200, ge=1)


class _DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    module_factory: str = "game_cls.data.module:build_game_video_pair_data_module"
    index_codec: _IndexCodecSelector
    backend: _BackendSelector


# ----------------------------------------------------------------------- #
# Legacy section models (strict — ``extra="forbid"`` catches typos like
# ``local_batch_szie`` or ``learning_ratae`` that would otherwise be silently
# ignored, causing hard-to-debug training behavior).
# ----------------------------------------------------------------------- #
class _ExperimentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    seed: int = 42
    output_dir: str = "runs/default"


class _DeviceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accelerator: str = "cpu"
    amp: bool = False
    amp_dtype: str = "bfloat16"


class _ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factory: str = ""
    checkpoint_path: str | None = None
    trainable_name_contains: str = "cls"
    num_classes: int = Field(default=2, ge=2)
    freeze_batchnorm_stats: bool | None = None
    freeze_backbone_batchnorm_stats: bool = True
    freeze_cls_batchnorm_stats: bool = True
    require_pretrained_backbone: bool = True


class _LossSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cross_entropy_weight: float = Field(default=1.0, ge=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold_loss_weight: float = Field(default=0.2, ge=0.0)
    threshold_safety_margin: float = Field(default=0.2, ge=0.0)
    threshold_temperature: float = Field(default=0.5, ge=0.0)
    threshold_warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    threshold_ramp_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    pos_weight: float | None = None


class _OptimizerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "AdamW"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)


class _SchedulerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    min_learning_rate: float = Field(default=0.0, ge=0.0)
    total_steps: int | None = None


class _TrainSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=1, ge=1)
    steps_per_epoch: int = Field(default=100, ge=1)
    max_steps: int | None = None
    log_every_steps: int = Field(default=10, ge=1)
    gradient_clip_norm: float = Field(default=5.0, ge=0.0)
    resume_path: str | None = None
    stop_after_steps: int | None = None
    verify_frozen_parameters: bool = False


class _DataLoaderSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    multiprocessing_context: str | None = None
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = True
    worker_num_threads: int = Field(default=1, ge=1)
    train: dict[str, Any] = Field(default_factory=dict)
    eval: dict[str, Any] = Field(default_factory=dict)


class _CheckpointSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_last_every_steps: int = Field(default=0, ge=0)
    save_best_selection: bool = True
    periodic_state_mode: str = "full"
    full_model_every_steps: int = Field(default=0, ge=0)


class _EvaluationLegacySection(BaseModel):
    """Legacy flat evaluation keys (threshold, amp, selection_metric, ...)."""
    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    amp: bool = False
    amp_dtype: str = "bfloat16"
    full_auc_mode: str = "histogram"
    auc_histogram_bins: int = Field(default=4096, ge=1)
    quick_save_error_limit: int = Field(default=200, ge=0)
    quick_test_pairs_per_video: int = Field(default=128, ge=1)
    parquet_row_group_size: int = Field(default=4096, ge=1)
    selection_metric: str = "global_f1_tau099"
    minimum_worst_game_f1: float | None = None
    selection_weights: dict[str, float] = Field(default_factory=dict)
    quick_test_every_steps: int = Field(default=0, ge=0)
    full_test_every_steps: int = Field(default=0, ge=0)
    full_test_at_end: bool = True
    html_max_errors_per_group: int = Field(default=200, ge=1)


class ExperimentConfig(BaseModel):
    """V2 experiment configuration.

    The new *component-selector* sections (task, trainable, data, sampler,
    runtime, evaluation) are strict: an unknown key inside them is rejected so
    that a typo in an extensible component fails fast. Legacy top-level sections
    (experiment, device, model, loss, train, dataloader, checkpoint, ...) are
    now also strict with ``extra="forbid"`` so that typos like
    ``local_batch_szie`` or ``learning_ratae`` fail fast instead of being
    silently ignored (USERPLAN §8.4, §14).
    """

    model_config = ConfigDict(extra="allow")

    config_version: Literal[2] = 2
    task: _TaskSelector
    trainable: _TrainableConfig
    data: _DataConfig
    sampler: _SamplerConfig
    runtime: _RuntimeConfig
    evaluation: _EvaluationConfig
    experiment: _ExperimentSection = Field(default_factory=_ExperimentSection)
    device: _DeviceSection = Field(default_factory=_DeviceSection)
    model: _ModelSection = Field(default_factory=_ModelSection)
    loss: _LossSection = Field(default_factory=_LossSection)
    optimizer: _OptimizerSection = Field(default_factory=_OptimizerSection)
    scheduler: _SchedulerSection = Field(default_factory=_SchedulerSection)
    train: _TrainSection = Field(default_factory=_TrainSection)
    dataloader: _DataLoaderSection = Field(default_factory=_DataLoaderSection)
    checkpoint: _CheckpointSection = Field(default_factory=_CheckpointSection)


def validate_and_normalize_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate a raw config mapping against the V2 schema.

    The loader has already performed V1 migration and override application, so
    this is purely structural validation with friendly error messages. Unknown
    keys inside the strict *component selectors* are rejected; legacy top-level
    sections are tolerated during the migration period.
    """
    if not isinstance(raw, dict):
        raise ValidationError("Configuration must be a YAML mapping at the top level.")
    # Check for unknown keys inside the strict component selectors before
    # handing the whole mapping to Pydantic (which tolerates legacy top-level
    # keys via extra="allow").
    _check_selector_keys(raw)
    try:
        config = ExperimentConfig(**raw)
    except PydanticValidationError as exc:
        raise _wrap_pydantic_error(exc) from exc
    return config


def _check_selector_keys(raw: dict[str, Any]) -> None:
    """Reject unknown keys inside the strict component-selector sections."""
    selector_sections: list[tuple[Any, type[BaseModel], str]] = [
        (raw.get("task"), _TaskSelector, "task"),
        (raw.get("trainable", {}).get("policy"), _TrainableSelector, "trainable.policy"),
        (raw.get("sampler", {}).get("policy"), _SamplerSelector, "sampler.policy"),
        (raw.get("runtime", {}).get("accelerator"), _AcceleratorSelector, "runtime.accelerator"),
        (raw.get("runtime", {}).get("distributed"), _DistributedSelector, "runtime.distributed"),
        (raw.get("evaluation", {}).get("suite"), _SuiteSelector, "evaluation.suite"),
        (raw.get("evaluation", {}).get("decision"), _DecisionSelector, "evaluation.decision"),
        (raw.get("data", {}).get("index_codec"), _IndexCodecSelector, "data.index_codec"),
        (raw.get("data", {}).get("backend"), _BackendSelector, "data.backend"),
    ]
    problems: list[str] = []
    for value, model, path in selector_sections:
        if isinstance(value, dict):
            for unknown in _unknown_keys(value, model, path):
                problems.append(f"Unknown key in {unknown}")
    if problems:
        raise ValidationError(
            "Configuration contains unknown component-selector keys:\n  "
            + "\n  ".join(problems)
        )
    # Validate plugin params against their specific Pydantic models so that
    # a typo like ``threshhold`` (instead of ``threshold``) fails fast.
    _check_plugin_params(raw)


def _check_plugin_params(raw: dict[str, Any]) -> None:
    """Validate plugin ``params`` against their specific Pydantic models.

    For each component selector that has a registered params model, this
    checks that the params dict conforms to the model — rejecting unknown
    keys, type mismatches, and out-of-range values.
    """
    plugin_sections: list[tuple[str, str, dict[str, Any]]] = [
        ("task", "type", raw.get("task", {})),
        ("trainable", "type", (raw.get("trainable", {}) or {}).get("policy", {})),
        ("sampler", "type", (raw.get("sampler", {}) or {}).get("policy", {})),
        ("runtime", "type", (raw.get("runtime", {}) or {}).get("distributed", {})),
        ("evaluation", "type", (raw.get("evaluation", {}) or {}).get("decision", {})),
        ("data", "type", (raw.get("data", {}) or {}).get("backend", {})),
    ]
    problems: list[str] = []
    for section, type_key, selector in plugin_sections:
        if not isinstance(selector, dict):
            continue
        plugin_type = selector.get(type_key)
        if not plugin_type:
            continue
        model = _PLUGIN_PARAMS_MODELS.get((section, str(plugin_type)))
        if model is None:
            # No strict model registered for this plugin type; skip.
            continue
        params = selector.get("params") or {}
        if not isinstance(params, dict):
            continue
        try:
            model(**params)
        except PydanticValidationError as exc:
            for error in exc.errors():
                loc = ".".join(str(item) for item in error["loc"])
                problems.append(f"{section}.params.{loc}: {error['msg']}")
    if problems:
        raise ValidationError(
            "Configuration contains invalid plugin params:\n  "
            + "\n  ".join(problems)
        )
