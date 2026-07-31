from __future__ import annotations

from typing import Any

from .loader import apply_overrides, deep_merge, load_config
from .migrations import migrate_to_latest
from .plugin_validation import cross_validate
from .schema import (
    ExperimentConfig,
    ValidationError,
    validate_and_normalize_config,
)


def load_and_validate_config(
    path: str, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Production entry point: load → migrate → override → validate.

    This is the *only* config loader that should be used by training entry
    points. It runs the full V2 pipeline:

    1. :func:`load_config` reads YAML (without applying overrides yet).
    2. :func:`migrate_to_latest` normalizes V1 configs to V2.
    3. Overrides are applied **after** migration so that users can override
       the new component-selector fields (e.g. ``task.params.positive_class_index``,
       ``runtime.distributed.params.broadcast_buffers``).
    4. :func:`validate_and_normalize_config` validates against the Pydantic
       schema and rejects unknown component-selector keys.
    5. :func:`cross_validate` runs cross-field checks (threshold range,
       DDP/WORLD_SIZE consistency).
    """
    raw = load_config(path)
    migrated = migrate_to_latest(raw)
    overridden = apply_overrides(migrated, overrides or [])
    config = validate_and_normalize_config(overridden)
    cross_validate(overridden)
    return config


def legacy_runtime_view(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a migrated V2 config into the flat shape the old trainer expects.

    The V2 migration turns scalar config values (e.g. ``data.backend: png``)
    into component-selector dicts (``data.backend: {type: png, params: {}}``).
    The legacy ``_run_training_loop`` still reads the flat shape, so we
    synthesize a backwards-compatible view that:

    * Restores ``data.backend`` to its plain string.
    * Restores ``device.accelerator`` from ``runtime.accelerator.type``.
    * Restores ``distributed.enabled`` / ``distributed.backend`` from
      ``runtime.distributed``.
    * Restores ``model.trainable_name_contains`` from the trainable policy.
    * Restores ``evaluation.threshold`` from ``evaluation.decision.params``.

    The returned dict is a **copy**; the original V2 config is not mutated.
    """
    from copy import deepcopy

    legacy = deepcopy(config)

    # data.backend: {type: png, params: {}} → "png"
    backend = legacy.get("data", {}).get("backend")
    if isinstance(backend, dict):
        legacy["data"]["backend"] = str(backend.get("type", "png"))

    # runtime.accelerator.type → device.accelerator
    runtime = legacy.get("runtime", {})
    accelerator = runtime.get("accelerator", {})
    if isinstance(accelerator, dict):
        legacy.setdefault("device", {})["accelerator"] = accelerator.get("type", "cpu")

    # runtime.distributed → distributed.{enabled, backend}
    distributed = runtime.get("distributed", {})
    if isinstance(distributed, dict):
        distributed_type = distributed.get("type", "single_process")
        legacy.setdefault("distributed", {})["enabled"] = distributed_type == "ddp"
        distributed_params = distributed.get("params") or {}
        if "backend" in distributed_params:
            legacy["distributed"]["backend"] = distributed_params["backend"]

    # trainable.policy.params.token → model.trainable_name_contains
    trainable = legacy.get("trainable", {})
    policy = trainable.get("policy", {})
    if isinstance(policy, dict):
        params = policy.get("params") or {}
        if "token" in params:
            legacy.setdefault("model", {})["trainable_name_contains"] = params["token"]

    # evaluation.decision.params.threshold → evaluation.threshold
    evaluation = legacy.get("evaluation", {})
    decision = evaluation.get("decision", {})
    if isinstance(decision, dict):
        decision_params = decision.get("params") or {}
        if "threshold" in decision_params:
            legacy["evaluation"]["threshold"] = decision_params["threshold"]

    return legacy


__all__ = [
    "load_config",
    "deep_merge",
    "migrate_to_latest",
    "validate_and_normalize_config",
    "cross_validate",
    "load_and_validate_config",
    "legacy_runtime_view",
    "ExperimentConfig",
    "ValidationError",
]
