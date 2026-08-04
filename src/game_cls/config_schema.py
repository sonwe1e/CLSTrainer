"""Strict, typed configuration schema for CLSTrainer.

This module is the single source of truth for which configuration keys
exist, what type they accept and what they mean. It enforces three rules:

1. Unknown configuration keys raise an error (with a "did you mean"
   suggestion) instead of being silently ignored.
2. Removed/deprecated keys raise a migration hint instead of being
   silently accepted.
3. Business contracts that must stay consistent (the deployment decision
   threshold) are resolved from exactly one place: ``decision.threshold``.

The schema deliberately avoids a heavyweight dependency such as Hydra or
Pydantic; it is a small declarative registry that the config loader,
the CLI and the documentation generator all share.
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass, field
from typing import Any


class ConfigSchemaError(ValueError):
    """Raised when a configuration violates the schema."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Key:
    """One configuration leaf.

    ``kind`` is one of ``int``, ``float``, ``bool``, ``str``, ``list``,
    ``dict``, ``any``. ``nullable`` allows an explicit ``null``. ``choices``
    restricts string/number values to an enumeration.
    """

    kind: str
    description: str
    nullable: bool = False
    choices: tuple[Any, ...] | None = None
    legacy: bool = False


def _k(kind: str, description: str, **kwargs: Any) -> Key:
    return Key(kind=kind, description=description, **kwargs)


_DATALOADER_ROLE_KEYS: dict[str, Any] = {
    "num_workers": _k("int", "DataLoader worker processes for this role."),
    "persistent_workers": _k(
        "bool", "Keep workers alive between epochs/evaluations."
    ),
    "prefetch_factor": _k("int", "Batches prefetched per worker."),
    "pin_memory": _k(
        "bool",
        "Pin host memory before device transfer. Keep false until an A/B "
        "test proves it helps.",
    ),
}

# ---------------------------------------------------------------------------
# The schema itself. Every key consumed anywhere in the codebase must be
# registered here; anything else is rejected.
# ---------------------------------------------------------------------------
SCHEMA: dict[str, Any] = {
    "experiment": {
        "name": _k(
            "str",
            "Human readable run name; used in the unique run directory name.",
        ),
        "seed": _k("int", "Base RNG seed; each rank adds its rank id."),
        "output_dir": _k(
            "str",
            "Run output location. With run_mode=unique it is the runs ROOT: "
            "every start creates a fresh dated subdirectory underneath it.",
        ),
        "run_mode": _k(
            "str",
            "fixed: write directly into output_dir (legacy). unique: "
            "allocate an immutable timestamped run directory under output_dir.",
            choices=("fixed", "unique"),
        ),
    },
    "device": {
        "accelerator": _k(
            "str",
            "Training device family.",
            choices=("auto", "cpu", "cuda", "npu"),
        ),
        "amp": _k("bool", "Enable mixed precision training."),
        "amp_dtype": _k(
            "str", "Mixed precision dtype.", choices=("float16", "bfloat16")
        ),
    },
    "data": {
        "synthetic": _k(
            "bool", "Use the built-in synthetic dataset (smoke tests only)."
        ),
        "strict_audit": _k(
            "bool",
            "Require a passing dataset audit before creating DataLoaders.",
        ),
        "audit_path": _k("str", "Path to audit.json produced by audit_dataset."),
        "train_index": _k("str", "Train frame index parquet."),
        "test_index": _k("str", "Test frame index parquet."),
        "train_video_index": _k(
            "str", "Train video-level entries parquet (row-per-video)."
        ),
        "test_video_index": _k(
            "str", "Test video-level entries parquet (row-per-video)."
        ),
        "backend": _k(
            "str",
            "Frame storage backend.",
            choices=("png", "packed_uint8"),
        ),
        "train_packed_index": _k(
            "str", "Packed train shard index (packed_uint8 backend).",
            nullable=True,
        ),
        "test_packed_index": _k(
            "str", "Packed test shard index (packed_uint8 backend).",
            nullable=True,
        ),
        "train_packed_video_index": _k(
            "str", "Packed train integer video index.", nullable=True
        ),
        "test_packed_video_index": _k(
            "str", "Packed test integer video index.", nullable=True
        ),
        "packed_max_open_shards": _k(
            "int", "LRU limit of simultaneously memmapped packed shards."
        ),
        "width": _k("int", "Frame width in pixels (contract: 448)."),
        "height": _k("int", "Frame height in pixels (contract: 208)."),
        "channels": _k("int", "Frame channels (contract: 3)."),
        "frame_extensions": _k(
            "list", "File extensions accepted as frames during scanning."
        ),
        "ignore_directory_prefixes": _k(
            "list", "Directory name prefixes pruned during scanning."
        ),
        "ignore_directory_names": _k(
            "list", "Exact directory names pruned during scanning."
        ),
        "ignore_file_globs": _k(
            "list", "File globs ignored during scanning."
        ),
        "unexpected_nested_directory_severity": _k(
            "str",
            "Severity for unexpected nested directories.",
            choices=("info", "warning", "error"),
        ),
        "ignored_example_limit": _k(
            "int", "Max example paths kept per ignored-file category."
        ),
        "require_content_hash_audit": _k(
            "bool", "Audit must include SHA-256 content hashes."
        ),
        "require_unique_video_keys_across_splits": _k(
            "bool",
            "Treat cross-split duplicate two-digit video ids as fatal.",
        ),
        "minimum_pairs_per_game_label_delta": _k(
            "dict",
            "Minimum legal pairs per (game,label) for each delta, "
            "e.g. {2: 1}.",
        ),
        "duplicate_policy": {
            "same_label_cross_split": _k(
                "str", "Severity for same-content duplicates across splits.",
                choices=("info", "warning", "error"),
            ),
            "same_label_within_split": _k(
                "str", "Severity for same-content duplicates within a split.",
                choices=("info", "warning", "error"),
            ),
            "cross_label_same_content": _k(
                "str",
                "Severity when identical content carries different labels.",
                choices=("info", "warning", "error"),
            ),
            "same_basename": _k(
                "str", "Severity for filename-only collisions.",
                choices=("info", "warning", "error"),
            ),
        },
    },
    "pair": {
        "train_delta_probability": _k(
            "dict", "Training frame-delta sampling distribution, e.g. {2: 0.7}."
        ),
        "test_delta": _k(
            "int", "Evaluation pair delta (contract: 2)."
        ),
    },
    "sampler": {
        "game_alpha": _k(
            "float", "Dirichlet smoothing for per-game balancing."
        ),
        "class_probability": _k(
            "dict", "Label sampling probability, e.g. {0: 0.5, 1: 0.5}."
        ),
        "deduplicate_within_global_batch": _k(
            "bool", "Avoid repeating a video inside one global batch."
        ),
    },
    "augmentation": {
        "enabled": _k("bool", "Master switch for deterministic pair augmentation."),
        "random_affine": {
            "enabled": _k("bool", "Enable random affine perturbations."),
            "probability": _k("float", "Per-pair probability."),
            "degrees": _k("float", "Rotation range in degrees."),
            "translate": _k("list", "Relative translation range [x, y]."),
            "scale": _k("list", "Scale range [min, max]."),
            "shear": _k("list", "Shear range in degrees."),
            "interpolation": _k("str", "Resampling filter name."),
            "fill": _k("any", "Fill value for uncovered pixels."),
        },
        "color_jitter": {
            "enabled": _k("bool", "Enable color jitter."),
            "probability": _k("float", "Per-pair probability."),
            "brightness": _k("float", "Brightness jitter magnitude."),
            "contrast": _k("float", "Contrast jitter magnitude."),
            "saturation": _k("float", "Saturation jitter magnitude."),
            "hue": _k("float", "Hue jitter magnitude."),
        },
        "random_erasing": {
            "enabled": _k("bool", "Enable random erasing."),
            "probability": _k("float", "Per-pair probability."),
            "scale": _k("list", "Erased area fraction range."),
            "ratio": _k("list", "Erased aspect ratio range."),
            "value": _k("any", "Fill value; 'random' for noise."),
        },
    },
    "model": {
        "factory": _k(
            "str",
            "Model factory 'package.module:function' returning a module "
            "that maps (image0, image1) to [B,2].",
        ),
        "checkpoint_path": _k(
            "str", "Base checkpoint; all non-cls weights must load from it.",
            nullable=True,
        ),
        "trainable_name_contains": _k(
            "str", "Substring selecting trainable parameters (contract: cls)."
        ),
        "num_classes": _k("int", "Output classes (contract: 2)."),
        "freeze_backbone_batchnorm_stats": _k(
            "bool", "Keep backbone BatchNorm statistics frozen."
        ),
        "freeze_cls_batchnorm_stats": _k(
            "bool",
            "Keep cls-head BatchNorm statistics frozen (required for "
            "distributed training without SyncBatchNorm).",
        ),
        "freeze_batchnorm_stats": _k(
            "bool",
            "Legacy global BatchNorm freeze switch; prefer the two "
            "role-specific keys above.",
            legacy=True,
        ),
        "require_pretrained_backbone": _k(
            "bool",
            "Fail unless every non-cls weight is fully loaded from the "
            "base checkpoint.",
        ),
    },
    "decision": {
        "threshold": _k(
            "float",
            "THE business decision threshold. Single source of truth: the "
            "training threshold loss, the evaluator and deployment all read "
            "this value. Strictly-greater-than this softmax probability "
            "means positive.",
        ),
    },
    "loss": {
        "cross_entropy_weight": _k("float", "Weight of the CE component."),
        "threshold": _k(
            "float",
            "Legacy copy of decision.threshold. Prefer decision.threshold; "
            "conflicting values are rejected.",
            legacy=True,
        ),
        "threshold_loss_weight": _k(
            "float", "Maximum weight of the threshold margin loss."
        ),
        "threshold_safety_margin": _k(
            "float", "Extra logit margin pushed beyond the decision boundary."
        ),
        "threshold_temperature": _k(
            "float", "Softplus temperature of the margin loss."
        ),
        "threshold_warmup_ratio": _k(
            "float", "Fraction of training before the margin loss activates."
        ),
        "threshold_ramp_ratio": _k(
            "float", "Fraction of training used to ramp the margin weight."
        ),
    },
    "optimizer": {
        "learning_rate": _k("float", "Peak AdamW learning rate."),
        "weight_decay": _k("float", "AdamW weight decay."),
    },
    "scheduler": {
        "warmup_steps": _k("int", "Linear warmup steps for the cosine schedule."),
        "min_learning_rate": _k("float", "Cosine schedule floor."),
    },
    "train": {
        "epochs": _k("int", "Epoch count (used when max_steps is null)."),
        "steps_per_epoch": _k("int", "Sampler steps per epoch."),
        "max_steps": _k(
            "int", "Global step budget; overrides epochs when set.",
            nullable=True,
        ),
        "stop_after_steps": _k(
            "int", "Optional early stop for staged acceptance runs.",
            nullable=True,
        ),
        "resume_path": _k(
            "str", "Checkpoint used for exact resume.", nullable=True
        ),
        "verify_frozen_parameters": _k(
            "bool", "Assert frozen weights stay bitwise unchanged."
        ),
        "local_batch_size": _k("int", "Per-rank batch size."),
        "gradient_clip_norm": _k("float", "Global gradient clip norm."),
        "log_every_steps": _k("int", "Metric/log cadence in steps."),
    },
    "dataloader": {
        "num_workers": _k(
            "int",
            "Fallback worker count; role-specific dataloader.train/eval "
            "values win when present.",
            legacy=True,
        ),
        "persistent_workers": _k(
            "bool", "Fallback persistent_workers.", legacy=True
        ),
        "prefetch_factor": _k("int", "Fallback prefetch_factor.", legacy=True),
        "pin_memory": _k("bool", "Fallback pin_memory.", legacy=True),
        "multiprocessing_context": _k(
            "str",
            "Worker start method; NPU forces spawn.",
            nullable=True,
            choices=("spawn", "fork", "forkserver"),
        ),
        "timeout_seconds": _k(
            "float", "Max wait per batch before a DataLoader timeout."
        ),
        "worker_num_threads": _k(
            "int", "CPU threads allowed inside each worker process."
        ),
        "train": dict(_DATALOADER_ROLE_KEYS),
        "eval": dict(_DATALOADER_ROLE_KEYS),
    },
    "evaluation": {
        "threshold": _k(
            "float",
            "Legacy copy of decision.threshold. Prefer decision.threshold; "
            "conflicting values are rejected.",
            legacy=True,
        ),
        "amp": _k("bool", "Mixed precision for evaluation forward passes."),
        "amp_dtype": _k(
            "str", "Evaluation AMP dtype (deployment parity).",
            choices=("float16", "bfloat16"),
        ),
        "quick_test_every_steps": _k(
            "int", "Quick-test cadence; 0 disables."
        ),
        "quick_test_pairs_per_video": _k(
            "int", "Quick-test pairs sampled per video."
        ),
        "full_test_every_steps": _k(
            "int", "Full-test cadence (observed dev-test); 0 disables."
        ),
        "full_test_at_end": _k("bool", "Run a final full test after training."),
        "full_auc_mode": _k(
            "str",
            "Distributed AUC strategy: fixed histogram or exact gather.",
            choices=("histogram", "exact"),
        ),
        "auc_histogram_bins": _k("int", "Histogram bins for AUC estimation."),
        "quick_save_error_limit": _k(
            "int", "Global cap on quick-test error exports."
        ),
        "html_max_errors_per_group": _k(
            "int", "Per-group error cap in the HTML report."
        ),
        "parquet_row_group_size": _k(
            "int", "Records accumulated before writing a parquet row group."
        ),
        "selection_metric": _k(
            "str",
            "Metric used to pick the best checkpoint.",
            choices=(
                "global_f1_tau099",
                "macro_game_f1_tau099",
                "worst_game_f1_tau099",
                "composite",
            ),
        ),
        "selection_weights": _k(
            "dict",
            "Component weights for composite selection, e.g. "
            "{global_f1: 0.4, macro_game_f1: 0.4, worst_game_f1: 0.2}.",
        ),
        "minimum_worst_game_f1": _k(
            "float",
            "Reject best-checkpoint candidates whose worst game F1 falls "
            "below this gate.",
            nullable=True,
        ),
    },
    "checkpoint": {
        "save_last_every_steps": _k(
            "int", "Periodic resume checkpoint cadence; 0 disables."
        ),
        "save_best_selection": _k(
            "bool", "Clone the best observed dev-test checkpoint."
        ),
        "save_best_test_f1": _k(
            "bool", "Legacy alias of save_best_selection.", legacy=True
        ),
        "periodic_state_mode": _k(
            "str",
            "Periodic checkpoint content: full state or trainable-only.",
            choices=("full", "trainable_only"),
        ),
        "full_model_every_steps": _k(
            "int", "Cadence for full-weight periodic saves; 0 disables."
        ),
    },
    "distributed": {
        "enabled": _k("bool", "Initialize c10d process groups."),
        "backend": _k(
            "str", "Process group backend, e.g. gloo / nccl / hccl."
        ),
    },
}

# Keys that used to exist but have no consumer anymore. They raise a
# migration hint instead of being silently accepted ("looks effective but
# is not" is worse than an error).
REMOVED_KEYS: dict[str, str] = {
    "optimizer.name": (
        "The optimizer is fixed to AdamW and was never selected by this "
        "field. Remove it."
    ),
    "scheduler.name": (
        "The scheduler is a fixed cosine-with-warmup implementation and was "
        "never selected by this field. Remove it."
    ),
    "evaluation.save_all_errors": (
        "Removed: quick-test error exports are bounded by "
        "evaluation.quick_save_error_limit, full tests always stream all "
        "errors to parquet shards."
    ),
}


def _type_matches(value: Any, key: Key) -> bool:
    if value is None:
        return key.nullable
    kind = key.kind
    if kind == "any":
        return True
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "list":
        return isinstance(value, list)
    if kind == "dict":
        return isinstance(value, dict)
    raise AssertionError(f"Unknown schema kind: {kind}")


def _suggest(unknown: str, candidates: list[str]) -> str:
    matches = difflib.get_close_matches(unknown, candidates, n=1, cutoff=0.6)
    if matches:
        return f" Did you mean '{matches[0]}'?"
    return ""


def _walk(
    node: Any,
    schema: dict[str, Any],
    path: str,
    problems: list[str],
) -> None:
    if not isinstance(node, dict):
        problems.append(f"{path or '<root>'}: expected a mapping.")
        return
    for key, value in node.items():
        dotted = f"{path}.{key}" if path else key
        if key not in schema:
            problems.append(
                f"Unknown config key: {dotted}."
                + _suggest(key, [str(k) for k in schema])
            )
            continue
        spec = schema[key]
        if isinstance(spec, dict):
            if isinstance(value, dict):
                _walk(value, spec, dotted, problems)
            else:
                problems.append(
                    f"{dotted}: expected a mapping with keys "
                    f"{sorted(spec)}, got {type(value).__name__}."
                )
            continue
        assert isinstance(spec, Key)
        if not _type_matches(value, spec):
            allowed = spec.kind + (" or null" if spec.nullable else "")
            problems.append(
                f"{dotted}: expected {allowed}, got "
                f"{type(value).__name__} ({value!r})."
            )
            continue
        if spec.choices is not None and value is not None:
            if value not in spec.choices:
                problems.append(
                    f"{dotted}: must be one of {sorted(map(str, spec.choices))}, "
                    f"got {value!r}."
                )


def validate_config(config: dict[str, Any]) -> None:
    """Raise ConfigSchemaError listing every unknown/mistyped key."""
    problems: list[str] = []
    _walk(config, SCHEMA, "", problems)
    if problems:
        raise ConfigSchemaError(problems)


def check_removed_keys(config: dict[str, Any]) -> None:
    problems: list[str] = []
    for dotted, hint in REMOVED_KEYS.items():
        cursor: Any = config
        found = True
        for part in dotted.split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                found = False
                break
        if found:
            problems.append(f"Removed config key: {dotted}. {hint}")
    if problems:
        raise ConfigSchemaError(problems)


def known_dotted_paths() -> list[str]:
    """Every schema-known leaf path, for override validation and docs."""
    paths: list[str] = []

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, spec in node.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(spec, dict):
                walk(spec, dotted)
            else:
                paths.append(dotted)

    walk(SCHEMA, "")
    return sorted(paths)


def check_override_path(dotted: str) -> None:
    """CLI overrides may only address schema-known keys."""
    if dotted in REMOVED_KEYS:
        raise ConfigSchemaError(
            [f"Removed config key: {dotted}. {REMOVED_KEYS[dotted]}"]
        )
    if dotted in known_dotted_paths():
        return
    suggestion = difflib.get_close_matches(
        dotted, known_dotted_paths(), n=1, cutoff=0.6
    )
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise ConfigSchemaError(
        [
            f"Unknown config override target: {dotted}.{hint} "
            "Overrides may only set keys that exist in the configuration "
            "schema; see 'cls-trainer config reference'."
        ]
    )


def resolve_decision_threshold(config: dict[str, Any]) -> float:
    """Merge the single business threshold into loss and evaluation.

    ``decision.threshold`` is the source of truth. Legacy ``loss.threshold``
    and ``evaluation.threshold`` are tolerated only while they agree with
    it; diverging values are a configuration error.
    """
    decision_cfg = config.get("decision") or {}
    candidates = {
        "decision.threshold": decision_cfg.get("threshold"),
        "loss.threshold": (config.get("loss") or {}).get("threshold"),
        "evaluation.threshold": (config.get("evaluation") or {}).get(
            "threshold"
        ),
    }
    provided = {
        name: float(value)
        for name, value in candidates.items()
        if value is not None
    }
    if not provided:
        threshold = 0.99
    else:
        distinct = set(provided.values())
        if len(distinct) > 1:
            detail = ", ".join(
                f"{name}={value}" for name, value in sorted(provided.items())
            )
            raise ConfigSchemaError(
                [
                    "Conflicting decision thresholds: "
                    f"{detail}. Keep a single source of truth in "
                    "decision.threshold."
                ]
            )
        threshold = next(iter(distinct))
    config.setdefault("decision", {})["threshold"] = threshold
    config.setdefault("loss", {})["threshold"] = threshold
    config.setdefault("evaluation", {})["threshold"] = threshold
    return threshold


def finalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a merged configuration.

    Returns a deep copy so callers' dicts are never mutated in place.
    Idempotent: finalizing an already-finalized config is a no-op.
    """
    finalized = copy.deepcopy(config)
    check_removed_keys(finalized)
    validate_config(finalized)
    resolve_decision_threshold(finalized)
    return finalized


def describe_reference() -> list[dict[str, Any]]:
    """Flat, documentation-ready listing of every schema key."""
    rows: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, spec in sorted(node.items()):
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(spec, dict):
                walk(spec, dotted)
                continue
            rows.append(
                {
                    "path": dotted,
                    "type": spec.kind + ("|null" if spec.nullable else ""),
                    "choices": list(spec.choices) if spec.choices else None,
                    "legacy": spec.legacy,
                    "description": spec.description,
                }
            )

    walk(SCHEMA, "")
    return rows
