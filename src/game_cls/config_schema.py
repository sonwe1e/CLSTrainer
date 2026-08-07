"""Strict, typed configuration schema for CLSTrainer.

This module is the single source of truth for which configuration keys
exist, what type they accept and what they mean. It enforces three rules:

1. Unknown configuration keys raise an error (with a "did you mean"
   suggestion) instead of being silently ignored.
2. Removed/deprecated keys raise a migration hint instead of being
   silently accepted.
3. Task-profile facts (the deployment decision threshold, frame size,
   pair delta, output width, trainable scope) are resolved from exactly
   one place: ``decision.threshold`` for the threshold; the rest are
   defaults that recipes may override — the framework must stay
   consistent with whatever the resolved config says.

The schema deliberately avoids a heavyweight dependency such as Hydra or
Pydantic; it is a small declarative registry that the config loader,
the CLI and the documentation generator all share.
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass
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
    "persistent_workers": _k("bool", "Keep workers alive between epochs/evaluations."),
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
        "smoke_mode": _k(
            "bool",
            "Mark a run as a smoke test so the production safety policy is "
            "relaxed (e.g. the require-a-full-validation-source gate). Real "
            "training must never set this; the NPU smoke stages disable full "
            "validation to probe forward/spawn/augmentation/eval separately "
            "(audit PR-E: test-env policy and production safety policy must "
            "not fight each other).",
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
        "val_index": _k(
            "str",
            "Validation frame index parquet. When omitted, test_index is "
            "aliased as validation and the run has no independent test set.",
            nullable=True,
        ),
        "test_index": _k("str", "Test frame index parquet."),
        "train_video_index": _k(
            "str", "Train video-level entries parquet (row-per-video)."
        ),
        "val_video_index": _k(
            "str",
            "Validation video-level entries parquet (row-per-video).",
            nullable=True,
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
            "str",
            "Packed train shard index (packed_uint8 backend).",
            nullable=True,
        ),
        "val_packed_index": _k(
            "str",
            "Packed validation shard index (packed_uint8 backend).",
            nullable=True,
        ),
        "test_packed_index": _k(
            "str",
            "Packed test shard index (packed_uint8 backend).",
            nullable=True,
        ),
        "train_packed_video_index": _k(
            "str", "Packed train integer video index.", nullable=True
        ),
        "val_packed_video_index": _k(
            "str", "Packed validation integer video index.", nullable=True
        ),
        "test_packed_video_index": _k(
            "str", "Packed test integer video index.", nullable=True
        ),
        "packed_max_open_shards": _k(
            "int", "LRU limit of simultaneously memmapped packed shards."
        ),
        "width": _k("int", "Frame width in pixels (task profile default: 448)."),
        "height": _k("int", "Frame height in pixels (task profile default: 208)."),
        "channels": _k("int", "Frame channels (task profile default: 3)."),
        "frame_extensions": _k(
            "list", "File extensions accepted as frames during scanning."
        ),
        "ignore_directory_prefixes": _k(
            "list", "Directory name prefixes pruned during scanning."
        ),
        "ignore_directory_names": _k(
            "list", "Exact directory names pruned during scanning."
        ),
        "ignore_file_globs": _k("list", "File globs ignored during scanning."),
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
        "require_independent_test": _k(
            "bool",
            "Production acceptance gate: refuse to train when the test "
            "split is aliased as the validation split.",
        ),
        "split_migration": _k(
            "dict",
            "Auto-generated notes describing legacy split aliasing "
            "(set by finalize_config; do not configure manually).",
        ),
        "source_root": _k(
            "str",
            "Single root scanned for both train and validation "
            "(split.mode=from_train).",
            nullable=True,
        ),
        "test_root": _k(
            "str",
            "Independent test root used with split.mode=from_train.",
            nullable=True,
        ),
        "prepare_if_missing": _k(
            "bool",
            "Auto-run dataset prepare before training when split indexes are missing.",
        ),
        "split": _k(
            "dict",
            "Source-video-level train/validation split configuration (step4 §二).",
        ),
        "source_video_identity": _k(
            "dict",
            "Source video identity contract (step7): how a source video uid "
            "is derived. game_video (default) treats video_id as unique per "
            "game; game_label_video treats video_id as unique per "
            "(game,label) and prints a leakage warning.",
        ),
        "minimum_pairs_per_game_label_delta": _k(
            "dict",
            "Minimum legal pairs per (game,label) for each delta, e.g. {2: 1}.",
        ),
        "deduplication": {
            "level": _k(
                "str",
                "Within one global batch: 'pair' avoids identical "
                "(video, delta, start) triples; 'video' avoids repeating "
                "the same video at all; 'none' samples freely.",
                choices=("none", "pair", "video"),
            ),
            "on_exhaustion": _k(
                "str",
                "What to do when a global batch cannot be filled without "
                "repeating the dedup identity: 'error' fails the run, "
                "'warn_and_relax' logs a warning and relaxes the "
                "constraint for that batch.",
                choices=("error", "warn_and_relax"),
            ),
        },
        "duplicate_policy": {
            "same_label_cross_split": _k(
                "str",
                "Severity for same-content duplicates across splits.",
                choices=("info", "warning", "error"),
            ),
            "same_label_within_split": _k(
                "str",
                "Severity for same-content duplicates within a split.",
                choices=("info", "warning", "error"),
            ),
            "cross_label_same_content": _k(
                "str",
                "Severity when identical content carries different labels.",
                choices=("info", "warning", "error"),
            ),
            "same_basename": _k(
                "str",
                "Severity for filename-only collisions.",
                choices=("info", "warning", "error"),
            ),
        },
        "metadata_sidecar": _k(
            "str",
            "Optional per-video metadata parquet keyed by source_video_uid "
            "(negative_subtype, sample_weight). Joined AFTER the split; "
            "never part of split/dedup identity (step5 P2).",
            nullable=True,
        ),
        "hard_negative": {
            "enabled": _k(
                "bool",
                "Mix ordinary and hard negative videos by subtype bucket "
                "during sampling (requires metadata_sidecar).",
            ),
            "subtype_field": _k(
                "str",
                "VideoEntry attribute holding the subtype label.",
            ),
            "hard_subtypes": _k("list", "Subtype values treated as hard negatives."),
            "ordinary_subtypes": _k(
                "list",
                "Subtype values treated as ordinary negatives; empty means "
                "every subtype not listed in hard_subtypes.",
            ),
            "negative_mix": _k(
                "dict",
                "Sampling weights per bucket, e.g. {ordinary: 0.5, hard: 0.5}.",
            ),
            "max_pairs_per_video": _k(
                "int",
                "Optional cap on how many start positions each video "
                "contributes (deterministic first N).",
                nullable=True,
            ),
            "min_videos_per_subtype_bucket": _k(
                "int",
                "Minimum eligible videos required in a bucket before it is "
                "used; smaller buckets fall back to the other bucket.",
            ),
        },
        "mining": {
            "enabled": _k(
                "bool",
                "Enable the hard-negative mining workflow (scan-negatives "
                "and --from-mining annotation).",
            ),
            "pool_index": _k(
                "str",
                "Frame index of the training-side negative pool to scan.",
                nullable=True,
            ),
            "pool_video_index": _k(
                "str",
                "Video-level index of the mining pool.",
                nullable=True,
            ),
            "pool_metadata": _k(
                "str",
                "Optional sidecar of the mining pool (subtype_before).",
                nullable=True,
            ),
            "pool_packed_index": _k(
                "str",
                "packed_uint8 shard index of the mining pool; required when "
                "data.backend=packed_uint8.",
                nullable=True,
            ),
            "pool_packed_video_index": _k(
                "str",
                "Video-level index of the packed mining pool; falls back to "
                "pool_video_index when unset.",
                nullable=True,
            ),
            "output": _k("str", "Output hard_negatives.parquet mining manifest path."),
            "top_k_per_video": _k(
                "int",
                "Max negatives kept per source video (avoids continuous "
                "frames drowning the manifest).",
            ),
            "max_samples": _k(
                "int", "Optional global cap on mined samples.", nullable=True
            ),
            "score_threshold": _k(
                "float",
                "Optional p_positive floor; only negatives at or above it are kept.",
                nullable=True,
            ),
            "version": _k("int", "Mining manifest format version."),
        },
        "challenge_index": _k(
            "str",
            "Fixed challenge-set frame index (never part of train/val/test; "
            "only consumed by benchmark evaluate).",
            nullable=True,
        ),
        "challenge_video_index": _k(
            "str", "Challenge-set video-level index.", nullable=True
        ),
        "challenge_metadata": _k(
            "str", "Optional challenge-set metadata sidecar.", nullable=True
        ),
        "challenge_packed_index": _k(
            "str",
            "packed_uint8 shard index of the challenge set; required when "
            "data.backend=packed_uint8.",
            nullable=True,
        ),
        "challenge_packed_video_index": _k(
            "str",
            "Video-level index of the packed challenge set; falls back to "
            "challenge_video_index when unset.",
            nullable=True,
        ),
    },
    "pair": {
        "train_delta_probability": _k(
            "dict", "Training frame-delta sampling distribution, e.g. {2: 0.7}."
        ),
        "test_delta": _k("int", "Evaluation pair delta (task profile default: 2)."),
    },
    "sampler": {
        "game_alpha": _k("float", "Dirichlet smoothing for per-game balancing."),
        "class_probability": _k(
            "dict", "Label sampling probability, e.g. {0: 0.5, 1: 0.5}."
        ),
        "deduplicate_within_global_batch": _k(
            "bool",
            "Legacy alias of data.deduplication.level: true=pair, false=none.",
            legacy=True,
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
        "random_perspective": {
            "enabled": _k("bool", "Enable random perspective warp."),
            "probability": _k("float", "Per-pair probability."),
            "distortion_scale": _k(
                "float", "Perspective distortion strength, typically <= 1."
            ),
        },
        "random_resized_crop": {
            "enabled": _k("bool", "Enable random resized crop."),
            "probability": _k("float", "Per-pair probability."),
            "scale": _k("list", "Crop area fraction range [min, max]."),
            "ratio": _k("list", "Crop aspect-ratio range [min, max]."),
            "size": _k("list", "Output size [height, width]."),
        },
        "gamma": {
            "enabled": _k("bool", "Enable gamma correction."),
            "probability": _k("float", "Per-pair probability."),
            "gamma_range": _k("list", "Gamma range [min, max]."),
        },
        "exposure": {
            "enabled": _k("bool", "Enable exposure adjustment."),
            "probability": _k("float", "Per-pair probability."),
            "factor_range": _k("list", "Exposure factor range [min, max]."),
        },
        "blur": {
            "enabled": _k("bool", "Enable Gaussian blur."),
            "probability": _k("float", "Per-pair probability."),
            "kernel_size": _k("int", "Odd Gaussian blur kernel size."),
            "sigma_range": _k("list", "Gaussian sigma range [min, max]."),
        },
        "noise": {
            "enabled": _k("bool", "Enable Gaussian noise injection."),
            "probability": _k("float", "Per-pair probability."),
            "noise_std": _k("float", "Additive noise standard deviation."),
        },
        "jpeg_compression": {
            "enabled": _k("bool", "Enable JPEG re-encode artifact injection."),
            "probability": _k("float", "Per-pair probability."),
            "quality_range": _k("list", "JPEG quality range [min, max]."),
        },
    },
    "model": {
        "factory": _k(
            "str",
            "Model factory 'package.module:function' returning a module "
            "that maps (image0, image1) to [B,2].",
        ),
        "checkpoint_path": _k(
            "str",
            "Base checkpoint; all non-cls weights must load from it.",
            nullable=True,
        ),
        "trainable_name_contains": _k(
            "str",
            "Substring selecting trainable parameters (task profile default: cls).",
        ),
        "trainable_rules": _k(
            "dict",
            "Staged partial unfreeze: dict keyed by rule name, each rule "
            "{pattern, lr_scale, unfreeze_at_step, priority}. When absent, "
            "trainable_name_contains is used (legacy behavior).",
        ),
        "num_classes": _k("int", "Output classes (task profile default: 2)."),
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
        "kwargs": _k(
            "dict",
            "Free-form project-specific factory arguments (e.g. "
            "cls_dropout). Passed to the model factory inside the model "
            "config mapping.",
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
        "threshold_warmup_steps": _k(
            "int",
            "Explicit step count before the margin loss activates. "
            "Overrides threshold_warmup_ratio when set; keeps the "
            "schedule independent of the total step budget.",
            nullable=True,
        ),
        "threshold_ramp_steps": _k(
            "int",
            "Explicit step count used to ramp the margin weight. "
            "Overrides threshold_ramp_ratio when set.",
            nullable=True,
        ),
        "label_smoothing": _k(
            "float",
            "Cross-entropy label smoothing. 0.0 keeps legacy behavior; "
            "keep small while the deployment threshold is fixed at 0.99.",
        ),
        "negative_tail_loss_weight": _k(
            "float",
            "Weight of the negative-tail OHEM component; 0 disables.",
        ),
        "negative_tail_hard_negative_k": _k(
            "int",
            "Top-k hardest negatives for the tail OHEM; null uses all negatives.",
            nullable=True,
        ),
        "rank_loss_weight": _k(
            "float",
            "Weight of the positive-vs-hard-negative pairwise ranking "
            "component; 0 disables.",
        ),
        "rank_margin": _k(
            "float",
            "Required logit margin between a positive and a hard negative "
            "in the ranking loss.",
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
            "int",
            "Global step budget; overrides epochs when set.",
            nullable=True,
        ),
        "stop_after_steps": _k(
            "int",
            "Optional early stop for staged acceptance runs.",
            nullable=True,
        ),
        "resume_path": _k("str", "Checkpoint used for exact resume.", nullable=True),
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
        "persistent_workers": _k("bool", "Fallback persistent_workers.", legacy=True),
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
            "str",
            "Evaluation AMP dtype (deployment parity).",
            choices=("float16", "bfloat16"),
        ),
        "quick_test_every_steps": _k(
            "int",
            "Legacy alias of val_quick_every_steps; migrated automatically.",
            legacy=True,
        ),
        "quick_test_pairs_per_video": _k(
            "int",
            "Legacy alias of val_quick_pairs_per_video; migrated automatically.",
            legacy=True,
        ),
        "full_test_every_steps": _k(
            "int",
            "Legacy alias of val_full_every_steps; migrated automatically.",
            legacy=True,
        ),
        "full_test_at_end": _k(
            "bool",
            "Legacy alias of val_full_at_end; migrated automatically.",
            legacy=True,
        ),
        "train_probe_every_steps": _k(
            "int",
            "Train-probe cadence (augmentation-free train subset evaluated "
            "like validation); 0 disables.",
        ),
        "train_probe_pairs_per_video": _k(
            "int", "Train-probe pairs sampled per train video."
        ),
        "val_quick_every_steps": _k(
            "int",
            "Quick validation cadence (fixed validation subset, high "
            "frequency trend watching); 0 disables.",
        ),
        "val_quick_pairs_per_video": _k(
            "int", "Quick validation pairs sampled per validation video."
        ),
        "val_full_every_steps": _k(
            "int",
            "Full validation cadence (drives model selection and early "
            "stopping); 0 disables.",
        ),
        "val_full_at_end": _k("bool", "Run a final full validation after training."),
        "tensorboard_live": _k(
            "bool",
            "Write TensorBoard scalars during training when the "
            "tensorboard package is importable; 0-cost when absent.",
        ),
        "full_auc_mode": _k(
            "str",
            "Distributed AUC strategy: fixed histogram or exact gather.",
            choices=("histogram", "exact"),
        ),
        "auc_histogram_bins": _k("int", "Histogram bins for AUC estimation."),
        "quick_save_error_limit": _k("int", "Global cap on quick-test error exports."),
        "html_max_errors_per_group": _k(
            "int", "Per-group error cap in the HTML report."
        ),
        "parquet_row_group_size": _k(
            "int", "Records accumulated before writing a parquet row group."
        ),
        "selection_metric": _k(
            "str",
            "Metric used to pick the best checkpoint. Neutral names do not "
            "bake in a fixed decision threshold; the legacy _tau099 names "
            "are still accepted.",
            choices=(
                "global_f1_at_decision_threshold",
                "macro_game_f1_at_decision_threshold",
                "worst_game_f1_at_decision_threshold",
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
        "selection_mode": _k(
            "str",
            "Model-selection strategy: metric (single metric), composite "
            "(weighted F1), or constrained (FPR/recall gates then "
            "recall/worst-recall/p99.9 ranking).",
            choices=("metric", "composite", "constrained"),
        ),
        "max_global_fpr": _k(
            "float",
            "Constrained selection: reject candidates whose global FPR at "
            "the decision threshold exceeds this (null disables the gate).",
            nullable=True,
        ),
        "max_worst_game_fpr": _k(
            "float",
            "Constrained selection: reject candidates whose worst-game FPR "
            "exceeds this (null disables).",
            nullable=True,
        ),
        "min_positive_recall": _k(
            "float",
            "Constrained selection: require global positive recall at or "
            "above this (null disables).",
            nullable=True,
        ),
        "max_fpr_for_recall": _k(
            "float", "FPR bound for recall_at_max_fpr and low-FPR partial AUC."
        ),
        "tail_calibration_enabled": _k(
            "bool", "Compute ece_tail_95_100 in evaluation."
        ),
        "group_by_negative_subtype": _k(
            "bool",
            "Add a game_label_subtype group catalog to evaluation and "
            "compute worst-subtype FPR/recall metrics (requires sidecar "
            "metadata with non-null negative_subtype).",
        ),
        "max_worst_subtype_fpr": _k(
            "float",
            "Constrained selection: reject candidates whose worst "
            "negative-subtype FPR exceeds this (null disables).",
            nullable=True,
        ),
    },
    "early_stopping": {
        "enabled": _k(
            "bool",
            "Stop training when the monitored validation metric plateaus. "
            "train.max_steps remains a safety upper bound.",
        ),
        "monitor": _k(
            "str",
            "Metric or selection contract watched for improvement. "
            "`selection_score` (alias `selection`) follows the unified "
            "selection contract: improvement is judged by the same ordering "
            "as best-checkpoint selection; in constrained mode that is "
            "(global_positive_recall, worst_game_positive_recall, "
            "-negative_score_p999), with ineligible evaluations counting "
            "toward patience rather than resetting it. Any other value names "
            "one numeric metric and uses the plain `mode` comparison.",
            choices=(
                "selection_score",
                "selection",
                "cross_entropy",
                "objective_loss",
                "worst_game_f1_at_decision_threshold",
                "worst_game_f1_tau099",
            ),
        ),
        "mode": _k(
            "str",
            "max: higher monitor values are better; min: lower values. "
            "Applies only to non-selection monitors (any `monitor` other "
            "than `selection_score`/`selection`); `mode: min` combined with "
            "a selection monitor is a config error because the selection "
            "rank key is always bigger-is-better.",
            choices=("max", "min"),
        ),
        "full_validation_only": _k(
            "bool",
            "Only full-validation evaluations may update patience; quick "
            "subsets are too noisy for stop decisions.",
        ),
        "burn_in_steps": _k("int", "Never stop before this global step."),
        "patience_evaluations": _k(
            "int",
            "Consecutive non-improving full validations tolerated before stopping.",
        ),
        "min_delta": _k(
            "float",
            "Minimum improvement that counts as an improvement; smaller "
            "deltas increment the patience counter.",
        ),
        "restore_best": _k(
            "bool",
            "Reload the best-selection checkpoint weights before the run finishes.",
        ),
    },
    "checkpoint": {
        "save_last_every_steps": _k(
            "int", "Periodic resume checkpoint cadence; 0 disables."
        ),
        "save_best_selection": _k(
            "bool",
            "Clone the best validation selection-score checkpoint "
            "(model_best_selection.pth).",
        ),
        "save_best_val_loss": _k(
            "bool",
            "Clone the best validation loss checkpoint (model_best_val_loss.pth).",
        ),
        "save_best_worst_game": _k(
            "bool",
            "Clone the best worst-game-F1 checkpoint (model_best_worst_game.pth).",
        ),
        "save_best_test_f1": _k(
            "bool", "Legacy alias of save_best_selection.", legacy=True
        ),
        "save_topk": _k(
            "int",
            "Keep the top-N full-validation checkpoints ranked by "
            "checkpoint.topk_monitor (saved as model_topk_<step>.pth); "
            "0 disables. Enabling it forces a last-checkpoint save after "
            "every full validation so the topk snapshot always matches the "
            "evaluated weights.",
        ),
        "topk_monitor": _k(
            "str",
            "`selection_score` (alias `selection`) ranks topk checkpoints "
            "by the unified selection contract, the same ordering as "
            "best-checkpoint selection; ineligible checkpoints are admitted "
            "but ranked strictly below every eligible one. Any other value "
            "names one numeric metric: lower is better for `cross_entropy`, "
            "higher is better otherwise.",
            choices=(
                "selection_score",
                "selection",
                "cross_entropy",
                "worst_game_f1_at_decision_threshold",
            ),
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
        "backend": _k("str", "Process group backend, e.g. gloo / nccl / hccl."),
    },
    "benchmark": {
        "output_dir": _k("str", "Directory for benchmark reports and data probes."),
        "gate_metrics": _k(
            "dict",
            "Release/benchmark gates keyed by the metric name the evaluator "
            'emits, each {op: "<="|"<"|">="|">", value: <number>}, e.g. '
            '{global_fpr_at_decision_threshold: {op: "<=", value: 0.01}, '
            'global_positive_recall_at_decision_threshold: {op: ">=", '
            "value: 0.8}}. A bare scalar bound is still accepted and means "
            "the metric's natural bound (upper for FPR/ECE/Brier/loss/"
            "negative-score, lower for recall/F1/precision/specificity/"
            "accuracy); metrics with no documented direction (sample_count, "
            "threshold) require the explicit form. Unknown metric names or "
            "operators are config errors; unmet gates fail the command.",
        ),
    },
    "export": {
        "format": _k(
            "str",
            "Default export format (weights|onnx).",
            choices=("weights", "onnx"),
        ),
        "output_dir": _k("str", "Directory for exported artifacts."),
        "onnx_opset": _k("int", "ONNX opset version for --format onnx."),
        "verify_samples": _k(
            "int", "Random sample tensors used to verify ONNX vs PyTorch."
        ),
        "include_threshold": _k(
            "bool", "Embed decision.threshold in the exported manifest."
        ),
    },
}

# Nested leaves of a dict-typed key. ``data.split`` is a documented dict
# block; its children live here so check_override_path, validate_config and
# describe_reference treat them exactly like nested-dict sections.
_SPLIT_KEYS: dict[str, Any] = {
    "mode": _k(
        "str",
        "Split derivation mode: 'off' keeps the legacy three-root index "
        "layout; 'from_train' scans source_root once and derives train/val "
        "by source video.",
        choices=("off", "from_train"),
    ),
    "val_ratio": _k(
        "float",
        "Fraction of source-video legal pairs moved to validation; "
        "strictly between 0 and 1 when mode is from_train.",
    ),
    "seed": _k(
        "int",
        "Deterministic split seed; the same seed and data produce a "
        "byte-identical manifest.",
    ),
    "group_key": _k("str", "Split unit identity; must be 'source_video_uid'."),
    "stratify_by": _k("list", "Stratum fields for balancing, e.g. ['game', 'label']."),
    "balance_by": _k("str", "Balancing statistic; must be 'legal_pair_count'."),
    "target_delta": _k(
        "int", "Frame delta whose pair count drives balancing (1, 2 or 3)."
    ),
    "manifest": _k(
        "str",
        "Split manifest parquet path, relative to the index output dir "
        "(default: split_manifest.parquet).",
    ),
    "on_new_groups": _k(
        "str",
        "Behavior when the dataset fingerprint changes: 'error' refuses to "
        "silently re-shuffle, 'extend' keeps every existing assignment and "
        "places only the new source videos. A change to seed, val_ratio or "
        "target_delta is always an error regardless of this setting.",
        choices=("error", "extend"),
    ),
    "small_stratum_policy": _k(
        "str",
        "Behavior for strata with fewer than two source videos: 'error' "
        "rejects, 'warn' keeps the lone video in train.",
        choices=("error", "warn"),
    ),
}

# Dotted paths whose Key is a dict but still validate nested leaves.
_NESTED_KEY_SCHEMAS: dict[str, dict[str, Any]] = {
    "data.split": _SPLIT_KEYS,
    "data.source_video_identity": {
        "mode": _k(
            "str",
            "game_video: uid = game::video_id (default, conservative). "
            "game_label_video: uid = game::label::video_id -- assumes "
            "identical video_id under different labels are unrelated videos.",
            choices=("game_video", "game_label_video"),
        ),
        "namespaces": {
            "train": _k(
                "str",
                "Source-pool namespace for the train split (explicit form). "
                "Train and val are split from one physical pool, so train "
                "must equal val.",
            ),
            "val": _k(
                "str",
                "Source-pool namespace for the validation split; must equal train.",
            ),
            "test": _k(
                "str",
                "Source-pool namespace for the test split; must differ from "
                "the train-side namespace (a declaration that test is an "
                "independent raw-video pool).",
            ),
            "source": _k(
                "str",
                "Train-side pool namespace under the {source, test} shorthand "
                "(use with data.split.mode=from_train).",
            ),
        },
    },
}


def _nested_key_schema(dotted: str) -> dict[str, Any] | None:
    """Sub-key schema of a dict-typed leaf, if one is registered."""
    return _NESTED_KEY_SCHEMAS.get(dotted)


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
                f"{dotted}: expected {allowed}, got {type(value).__name__} ({value!r})."
            )
            continue
        if spec.choices is not None and value is not None and value not in spec.choices:
            problems.append(
                f"{dotted}: must be one of {sorted(map(str, spec.choices))}, "
                f"got {value!r}."
            )
            continue
        nested = _nested_key_schema(dotted)
        if nested is not None and isinstance(value, dict):
            _walk(value, nested, dotted, problems)


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
                nested = _nested_key_schema(dotted)
                if nested is not None:
                    walk(nested, dotted)

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
        "evaluation.threshold": (config.get("evaluation") or {}).get("threshold"),
    }
    provided = {
        name: float(value) for name, value in candidates.items() if value is not None
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


# Documented defaults for the augmentation block. An absent block still
# yields the full dict so consumers can read every key unconditionally; every
# transform defaults to disabled so existing configs stay bit-identical (the
# consumer skips disabled transforms). ``_apply_defaults`` fills each key with
# ``setdefault``, so user-supplied values always win.
_DEFAULT_AUGMENTATION: dict[str, Any] = {
    "enabled": False,
    "random_affine": {
        "enabled": False,
        "probability": 0.0,
        "degrees": 0.0,
        "translate": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "shear": [0.0, 0.0],
        "interpolation": "bilinear",
        "fill": 0,
    },
    "color_jitter": {
        "enabled": False,
        "probability": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
    },
    "random_erasing": {
        "enabled": False,
        "probability": 0.0,
        "scale": [0.02, 0.33],
        "ratio": [0.3, 3.3],
        "value": 0,
    },
    "random_perspective": {
        "enabled": False,
        "probability": 0.0,
        "distortion_scale": 0.2,
    },
    "random_resized_crop": {
        "enabled": False,
        "probability": 0.0,
        "scale": [0.9, 1.0],
        "ratio": [0.9, 1.1],
        "size": [208, 448],
    },
    "gamma": {
        "enabled": False,
        "probability": 0.0,
        "gamma_range": [0.8, 1.2],
    },
    "exposure": {
        "enabled": False,
        "probability": 0.0,
        "factor_range": [0.85, 1.15],
    },
    "blur": {
        "enabled": False,
        "probability": 0.0,
        "kernel_size": 3,
        "sigma_range": [0.1, 1.0],
    },
    "noise": {
        "enabled": False,
        "probability": 0.0,
        "noise_std": 0.02,
    },
    "jpeg_compression": {
        "enabled": False,
        "probability": 0.0,
        "quality_range": [60, 95],
    },
}

# Defaults for the new loss keys. Every new weight defaults to 0.0 so existing
# configs (which never set them) keep their exact training curves; rank_margin
# and the null tail-k are the documented neutral values.
_DEFAULT_LOSS_NEW_KEYS: dict[str, Any] = {
    "negative_tail_loss_weight": 0.0,
    "negative_tail_hard_negative_k": None,
    "rank_loss_weight": 0.0,
    "rank_margin": 0.2,
}

# Hard-negative subtype mixing (step5 P2). All defaults keep the legacy
# sampler byte-identical when data.hard_negative.enabled is false.
_DEFAULT_HARD_NEGATIVE: dict[str, Any] = {
    "enabled": False,
    "subtype_field": "negative_subtype",
    "hard_subtypes": [],
    "ordinary_subtypes": [],
    "negative_mix": {"ordinary": 0.5, "hard": 0.5},
    "max_pairs_per_video": None,
    "min_videos_per_subtype_bucket": 1,
}

_DEFAULT_MINING: dict[str, Any] = {
    "enabled": False,
    "pool_index": None,
    "pool_video_index": None,
    "pool_metadata": None,
    "pool_packed_index": None,
    "pool_packed_video_index": None,
    "output": "indexes/hard_negatives.parquet",
    "top_k_per_video": 8,
    "max_samples": None,
    "score_threshold": None,
    "version": 1,
}


def _apply_defaults(config: dict[str, Any]) -> None:
    """Fill documented defaults for required keys a recipe may omit."""
    experiment = config.setdefault("experiment", {})
    if "seed" not in experiment:
        experiment["seed"] = 20260728
    if "name" not in experiment:
        experiment["name"] = "run"
    experiment.setdefault("smoke_mode", False)

    # An absent data.split block still yields the full dict so consumers can
    # rely on the documented defaults (split.mode=off keeps the legacy
    # three-root index layout).
    default_split = {
        "mode": "off",
        "val_ratio": 0.10,
        "seed": 20260728,
        "group_key": "source_video_uid",
        "stratify_by": ["game", "label"],
        "balance_by": "legal_pair_count",
        "target_delta": 2,
        "manifest": "split_manifest.parquet",
        "on_new_groups": "error",
        "small_stratum_policy": "error",
    }
    data = config.setdefault("data", {})
    data.setdefault("source_root", None)
    data.setdefault("test_root", None)
    data.setdefault("prepare_if_missing", False)
    data.setdefault("metadata_sidecar", None)
    hard_negative = data.setdefault("hard_negative", {})
    if not isinstance(hard_negative, dict):
        hard_negative = {}
        data["hard_negative"] = hard_negative
    for key, value in _DEFAULT_HARD_NEGATIVE.items():
        if isinstance(value, dict):
            block = hard_negative.setdefault(key, {})
            if isinstance(block, dict):
                for sub_key, sub_value in value.items():
                    block.setdefault(sub_key, sub_value)
        else:
            hard_negative.setdefault(key, value)
    mining = data.setdefault("mining", {})
    if not isinstance(mining, dict):
        mining = {}
        data["mining"] = mining
    for key, value in _DEFAULT_MINING.items():
        mining.setdefault(key, value)
    data.setdefault("challenge_index", None)
    data.setdefault("challenge_video_index", None)
    data.setdefault("challenge_metadata", None)
    data.setdefault("challenge_packed_index", None)
    data.setdefault("challenge_packed_video_index", None)
    benchmark = config.setdefault("benchmark", {})
    benchmark.setdefault("output_dir", "benchmarks")
    benchmark.setdefault("gate_metrics", {})
    export_cfg = config.setdefault("export", {})
    export_cfg.setdefault("format", "weights")
    export_cfg.setdefault("output_dir", "exports")
    export_cfg.setdefault("onnx_opset", 17)
    export_cfg.setdefault("verify_samples", 8)
    export_cfg.setdefault("include_threshold", True)
    data["split"] = {**default_split, **(data.get("split") or {})}
    data["source_video_identity"] = {
        **{"mode": "game_video"},
        **(data.get("source_video_identity") or {}),
    }

    evaluation = config.setdefault("evaluation", {})
    evaluation.setdefault("selection_mode", "metric")
    evaluation.setdefault("max_global_fpr", None)
    evaluation.setdefault("max_worst_game_fpr", None)
    evaluation.setdefault("min_positive_recall", None)
    evaluation.setdefault("max_fpr_for_recall", 0.01)
    evaluation.setdefault("tail_calibration_enabled", True)
    evaluation.setdefault("group_by_negative_subtype", False)
    evaluation.setdefault("max_worst_subtype_fpr", None)

    # An absent augmentation block still yields the full dict with every
    # transform disabled; an absent loss block still yields the new feature
    # weights at 0.0. setdefault never overwrites user-supplied values, so
    # existing configs are unchanged.
    augmentation = config.setdefault("augmentation", {})
    if not isinstance(augmentation, dict):
        augmentation = {}
        config["augmentation"] = augmentation
    for key, value in _DEFAULT_AUGMENTATION.items():
        if isinstance(value, dict):
            block = augmentation.setdefault(key, {})
            if isinstance(block, dict):
                for sub_key, sub_value in value.items():
                    block.setdefault(sub_key, sub_value)
        else:
            augmentation.setdefault(key, value)

    loss = config.setdefault("loss", {})
    if not isinstance(loss, dict):
        loss = {}
        config["loss"] = loss
    for key, value in _DEFAULT_LOSS_NEW_KEYS.items():
        loss.setdefault(key, value)


# Legacy evaluation cadence keys are silently migrated so old configs keep
# working; the new train/validation/test protocol names are canonical.
_LEGACY_EVALUATION_ALIASES: dict[str, str] = {
    "quick_test_every_steps": "val_quick_every_steps",
    "quick_test_pairs_per_video": "val_quick_pairs_per_video",
    "full_test_every_steps": "val_full_every_steps",
    "full_test_at_end": "val_full_at_end",
}


def migrate_split_roles(config: dict[str, Any]) -> None:
    """Normalize legacy two-split configs onto train/val/test roles.

    * When ``data.val_index`` is missing, ``data.test_index`` becomes the
      validation split and ``data.split_migration.test_used_as_validation``
      records that this run has no independent test set.
    * Legacy ``quick_test_*``/``full_test_*`` evaluation cadence keys are
      renamed to their ``val_*`` equivalents.

    Idempotent: finalizing an already-finalized config changes nothing.
    """
    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        migration = dict(data_cfg.get("split_migration") or {})
        if not data_cfg.get("val_index") and data_cfg.get("test_index"):
            data_cfg["val_index"] = data_cfg["test_index"]
            if data_cfg.get("test_video_index"):
                data_cfg["val_video_index"] = data_cfg["test_video_index"]
            if data_cfg.get("test_packed_index"):
                data_cfg["val_packed_index"] = data_cfg["test_packed_index"]
            if data_cfg.get("test_packed_video_index"):
                data_cfg["val_packed_video_index"] = data_cfg["test_packed_video_index"]
            migration["test_used_as_validation"] = True
        data_cfg["split_migration"] = migration
    evaluation_cfg = config.get("evaluation")
    if isinstance(evaluation_cfg, dict):
        for legacy_key, canonical_key in _LEGACY_EVALUATION_ALIASES.items():
            if legacy_key in evaluation_cfg and canonical_key not in evaluation_cfg:
                evaluation_cfg[canonical_key] = evaluation_cfg.pop(legacy_key)
            elif legacy_key in evaluation_cfg:
                evaluation_cfg.pop(legacy_key)


def split_role_warnings(config: dict[str, Any]) -> list[str]:
    """Human-readable warnings about legacy split aliasing."""
    warnings: list[str] = []
    migration = (config.get("data") or {}).get("split_migration") or {}
    if migration.get("test_used_as_validation"):
        warnings.append(
            "data.val_index is missing: test_index is used as the "
            "validation split. This run has NO independent test set; final "
            "test evaluation is unavailable until a val split is added."
        )
    return warnings


def resolve_source_identity_namespaces(
    source_video_identity: dict | None,
) -> dict[str, str]:
    """Normalize ``data.source_video_identity.namespaces`` to per-split roles.

    Returns ``{}`` when namespaces are unconfigured. Accepts the explicit
    ``{train, val, test}`` form or the ``{source, test}`` shorthand; both
    resolve to ``{train: <train-side>, val: <train-side>, test: <test>}``.
    Callers must run ``finalize_config`` first so the forms and equality
    constraints already hold.
    """
    raw = (source_video_identity or {}).get("namespaces")
    if not raw:
        return {}
    if "source" in raw:
        train_side = raw["source"]
        return {"train": train_side, "val": train_side, "test": raw["test"]}
    return {"train": raw["train"], "val": raw["val"], "test": raw["test"]}


def semantic_validate(config: dict[str, Any]) -> None:
    """Third validation layer: ranges, probabilities and cross-field facts.

    Structure (unknown keys/types) is checked by ``validate_config``; this
    layer rejects configurations that are well-typed but meaningless, such
    as non-positive batch sizes, negative learning rates, thresholds
    outside (0, 1), probability vectors that do not sum to one, or
    warmups longer than the whole training budget.
    """
    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    train_cfg = config.get("train") or {}
    if isinstance(train_cfg.get("local_batch_size"), (int, float)):
        require(
            train_cfg["local_batch_size"] > 0,
            "train.local_batch_size must be positive "
            f"(got {train_cfg['local_batch_size']}).",
        )
    if isinstance(train_cfg.get("steps_per_epoch"), (int, float)):
        require(
            train_cfg["steps_per_epoch"] > 0,
            "train.steps_per_epoch must be positive.",
        )
    if isinstance(train_cfg.get("log_every_steps"), (int, float)):
        require(
            train_cfg["log_every_steps"] > 0,
            "train.log_every_steps must be positive (0 would never log).",
        )
    if isinstance(train_cfg.get("epochs"), (int, float)):
        require(
            train_cfg["epochs"] > 0,
            "train.epochs must be positive.",
        )

    optimizer_cfg = config.get("optimizer") or {}
    for key in ("learning_rate", "weight_decay"):
        value = optimizer_cfg.get(key)
        if isinstance(value, (int, float)):
            require(
                value >= 0,
                f"optimizer.{key} must be non-negative (got {value}).",
            )

    decision = config.get("decision") or {}
    threshold = decision.get("threshold")
    if isinstance(threshold, (int, float)):
        require(
            0.0 < float(threshold) < 1.0,
            f"decision.threshold must be strictly between 0 and 1 (got {threshold}).",
        )

    loss_cfg = config.get("loss") or {}
    for key in ("threshold_temperature", "threshold_safety_margin"):
        value = loss_cfg.get(key)
        if isinstance(value, (int, float)):
            require(
                value > 0,
                f"loss.{key} must be positive (got {value}).",
            )
    for key in ("cross_entropy_weight", "threshold_loss_weight", "label_smoothing"):
        value = loss_cfg.get(key)
        if isinstance(value, (int, float)):
            require(
                value >= 0,
                f"loss.{key} must be non-negative (got {value}).",
            )
    # Stage 4 loss extensions: new weights are non-negative, the rank margin
    # is non-negative and the optional hard-negative tail cap is positive when
    # set (null means "use all negatives").
    for key in ("negative_tail_loss_weight", "rank_loss_weight"):
        value = loss_cfg.get(key)
        if isinstance(value, (int, float)):
            require(
                value >= 0,
                f"loss.{key} must be non-negative (got {value}).",
            )
    rank_margin = loss_cfg.get("rank_margin")
    if isinstance(rank_margin, (int, float)):
        require(
            rank_margin >= 0,
            f"loss.rank_margin must be non-negative (got {rank_margin}).",
        )
    tail_k = loss_cfg.get("negative_tail_hard_negative_k")
    if isinstance(tail_k, int):
        require(
            tail_k > 0,
            f"loss.negative_tail_hard_negative_k must be positive when set "
            f"(got {tail_k}); null uses all negatives.",
        )

    # Stage 4 augmentation extensions: per-transform probability in [0, 1] and
    # documented ranges as 2-element numeric lists with lower <= upper. Only
    # clearly invalid values are rejected; size is an output shape, so it only
    # has to be a 2-element int list (no ordering constraint).
    _AUGMENTATION_RANGES: dict[str, tuple[str, ...]] = {
        "random_perspective": (),
        "random_resized_crop": ("scale", "ratio"),
        "gamma": ("gamma_range",),
        "exposure": ("factor_range",),
        "blur": ("sigma_range",),
        "noise": (),
        "jpeg_compression": ("quality_range",),
    }
    augmentation_cfg = config.get("augmentation") or {}
    for name, range_keys in _AUGMENTATION_RANGES.items():
        block = augmentation_cfg.get(name)
        if not isinstance(block, dict):
            continue
        probability = block.get("probability")
        if isinstance(probability, (int, float)):
            require(
                0.0 <= float(probability) <= 1.0,
                f"augmentation.{name}.probability must be in [0, 1] "
                f"(got {probability}).",
            )
        for range_key in range_keys:
            value = block.get(range_key)
            if (
                isinstance(value, list)
                and len(value) == 2
                and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in value
                )
            ):
                require(
                    float(value[0]) <= float(value[1]),
                    f"augmentation.{name}.{range_key} must have lower <= "
                    f"upper (got {value}).",
                )
            elif value is not None:
                require(
                    False,
                    f"augmentation.{name}.{range_key} must be a 2-element "
                    f"numeric list (got {value!r}).",
                )
        size = block.get("size")
        if size is not None and (
            not isinstance(size, list)
            or len(size) != 2
            or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in size
            )
        ):
            require(
                False,
                f"augmentation.{name}.size must be a 2-element int list "
                f"(got {size!r}).",
            )

    evaluation_cfg = config.get("evaluation") or {}
    for key in (
        "train_probe_every_steps",
        "val_quick_every_steps",
        "val_full_every_steps",
        "train_probe_pairs_per_video",
        "val_quick_pairs_per_video",
    ):
        value = evaluation_cfg.get(key)
        if isinstance(value, (int, float)):
            require(
                value >= 0,
                f"evaluation.{key} must be non-negative (got {value}).",
            )
    weights = evaluation_cfg.get("selection_weights")
    if isinstance(weights, dict) and weights:
        require(
            all(
                isinstance(value, (int, float)) and value >= 0
                for value in weights.values()
            ),
            "evaluation.selection_weights must all be non-negative.",
        )
        require(
            sum(float(value) for value in weights.values()) > 0,
            "evaluation.selection_weights must sum to a positive value.",
        )
    # Audit P1-4: under selection_mode=constrained the rank key is
    # (global recall, worst-game recall, -negative p99.9) and the composite F1
    # weights can never influence the best checkpoint. A config carrying both
    # is dead-config noise: fail it instead of letting a user believe the
    # weights matter.
    if evaluation_cfg.get("selection_mode") == "constrained":
        require(
            evaluation_cfg.get("selection_metric") != "composite",
            "evaluation.selection_mode=constrained ignores a composite "
            "selection_metric; remove selection_metric or drop constrained mode.",
        )
        require(
            not evaluation_cfg.get("selection_weights"),
            "evaluation.selection_mode=constrained ignores "
            "evaluation.selection_weights; remove the weights or drop "
            "constrained mode.",
        )

    # Low-FPR evaluation protocol: FPR/recall bounds must be true
    # probabilities in (0, 1). Every gate is optional (null disables it), so
    # a constrained selection with all gates disabled degrades to pure recall
    # ranking and remains legal.
    for key in (
        "max_fpr_for_recall",
        "max_global_fpr",
        "max_worst_game_fpr",
        "min_positive_recall",
        "max_worst_subtype_fpr",
    ):
        value = evaluation_cfg.get(key)
        if value is not None and isinstance(value, (int, float)):
            require(
                0.0 < float(value) < 1.0,
                f"evaluation.{key} must be strictly between 0 and 1 (got {value}).",
            )

    # Benchmark gates (step6): metric names must be producible by the
    # evaluator and comparison operators must be real, so a typo fails here
    # instead of reading "metric absent" after a full benchmark run. The
    # import is local: config_schema is loaded very early and must not pull
    # in the reports package at module import time.
    from game_cls.reports.benchmark import validate_gate_metrics

    for problem in validate_gate_metrics(
        (config.get("benchmark") or {}).get("gate_metrics")
    ):
        require(False, problem)

    # Hard-negative subtype mixing (step5 P2): enabled requires a sidecar,
    # and the per-bucket weights must be positive.
    data_meta = config.get("data") or {}
    hard_negative = data_meta.get("hard_negative") or {}
    if hard_negative.get("enabled", False):
        require(
            bool(data_meta.get("metadata_sidecar")),
            "data.hard_negative.enabled requires data.metadata_sidecar to "
            "point at a per-video metadata parquet.",
        )
        # Without hard_subtypes every negative lands in the ordinary bucket
        # and the hard bucket is empty, so sampling silently degrades to
        # plain negative sampling. File existence and sidecar content are
        # checked at startup/`config validate` (see data.sidecar
        # .check_hard_negative_readiness); this layer stays filesystem-free.
        require(
            bool(hard_negative.get("hard_subtypes")),
            "data.hard_negative.enabled requires a non-empty "
            "data.hard_negative.hard_subtypes; otherwise no video can ever "
            "enter the hard bucket and sampling degrades to ordinary "
            "negatives.",
        )
        negative_mix = hard_negative.get("negative_mix") or {}
        mix_values = [
            negative_mix.get("ordinary", 0.0),
            negative_mix.get("hard", 0.0),
        ]
        require(
            any(float(value) > 0 for value in mix_values),
            "data.hard_negative.negative_mix must have a positive weight "
            "for at least one of {ordinary, hard}.",
        )
        max_pairs = hard_negative.get("max_pairs_per_video")
        if max_pairs is not None:
            require(
                int(max_pairs) >= 1,
                "data.hard_negative.max_pairs_per_video must be >= 1 when set.",
            )
        require(
            int(hard_negative.get("min_videos_per_subtype_bucket", 1)) >= 1,
            "data.hard_negative.min_videos_per_subtype_bucket must be >= 1.",
        )
    if evaluation_cfg.get("group_by_negative_subtype", False):
        # Subtype labels can arrive from the split sidecar (train/val/test) or
        # from an external pool's own sidecar (challenge / mining), so accept
        # any of the three rather than forcing a benchmark-only config to set
        # a train-side key it never reads.
        require(
            bool(
                data_meta.get("metadata_sidecar")
                or data_meta.get("challenge_metadata")
                or (data_meta.get("mining") or {}).get("pool_metadata")
            ),
            "evaluation.group_by_negative_subtype requires a per-video "
            "metadata sidecar so subtype labels are available: set "
            "data.metadata_sidecar, data.challenge_metadata, or "
            "data.mining.pool_metadata.",
        )

    # Probability vectors: non-negative and (approximately) sum to one.
    for key in ("class_probability", "delta_probability"):
        probabilities = (config.get("sampler") or {}).get(key)
        if not isinstance(probabilities, dict) or not probabilities:
            continue
        values = [float(item) for item in probabilities.values()]
        require(
            all(value >= 0 for value in values),
            f"sampler.{key} must not contain negative probabilities.",
        )
        if values and all(value == 0 for value in values):
            require(
                False,
                f"sampler.{key} is all zero; sampling would be meaningless.",
            )
        require(
            abs(sum(values) - 1.0) < 1e-3,
            f"sampler.{key} must sum to ~1 (got {sum(values):.4f}).",
        )

    # Cross-field: warmup must fit inside the total step budget. The public
    # field is scheduler.warmup_steps; train has no such key, and the old read
    # of train.warmup_steps was dead because _walk rejects the unknown key
    # before semantic validation runs (audit P1-5).
    scheduler_cfg = config.get("scheduler") or {}
    warmup = scheduler_cfg.get("warmup_steps")
    max_steps = train_cfg.get("max_steps")
    if isinstance(warmup, (int, float)) and isinstance(max_steps, (int, float)):
        require(
            warmup <= max_steps,
            f"scheduler.warmup_steps ({warmup}) must not exceed "
            f"train.max_steps ({max_steps}).",
        )

    # Auto split: from_train only makes sense with a usable val_ratio and a
    # supported target_delta. source_root is deliberately NOT required here:
    # the split indexes may already exist (prepare_if_missing=false).
    split_cfg = (config.get("data") or {}).get("split") or {}
    if split_cfg.get("mode") == "from_train":
        val_ratio = split_cfg.get("val_ratio")
        if isinstance(val_ratio, (int, float)):
            require(
                0.0 < float(val_ratio) < 1.0,
                f"data.split.val_ratio must be strictly between 0 and 1 "
                f"(got {val_ratio}).",
            )
        target_delta = split_cfg.get("target_delta")
        if isinstance(target_delta, (int, float)):
            require(
                int(target_delta) in (1, 2, 3),
                f"data.split.target_delta must be 1, 2 or 3 (got {target_delta}).",
            )

    # Source provenance namespaces (step8): per-split source-pool labels.
    # Namespaces exist only to separate independent collection pools that
    # coincidentally share local video_id numbering; the SHA-256 layer stays
    # namespace-blind, so identical content across splits is still fatal.
    svc = data_meta.get("source_video_identity") or {}
    namespaces = svc.get("namespaces")
    if namespaces is not None:
        ns_keys = set(namespaces)
        if "source" in ns_keys:
            # Only train/val are mutually exclusive with the source
            # shorthand; test is shared by both forms.
            require(
                not (ns_keys & {"train", "val"}),
                "data.source_video_identity.namespaces: the {source, test} "
                "shorthand is mutually exclusive with explicit train/val "
                "keys.",
            )
            require(
                "test" in ns_keys,
                "data.source_video_identity.namespaces: the {source, test} "
                "shorthand requires a test key.",
            )
        else:
            require(
                {"train", "val", "test"} <= ns_keys,
                "data.source_video_identity.namespaces: the explicit form "
                "requires train, val and test keys.",
            )
        # Only resolve once the configured form is complete; the resolver
        # indexes the missing keys directly, so guard before calling it.
        form_complete = (
            ("source" in ns_keys and "test" in ns_keys)
            if "source" in ns_keys
            else {"train", "val", "test"} <= ns_keys
        )
        resolved = resolve_source_identity_namespaces(svc) if form_complete else {}
        require(
            bool(resolved) and all(resolved.values()),
            "data.source_video_identity.namespaces values must be non-empty.",
        )
        if resolved:
            require(
                resolved["train"] == resolved["val"],
                "data.source_video_identity.namespaces: train and val must "
                "share one namespace because they are split from a single "
                "source pool.",
            )
            require(
                resolved["train"] != resolved["test"],
                "data.source_video_identity.namespaces: the train-side "
                "namespace must differ from the test namespace; identical "
                "namespaces would collapse the provenance boundary and hide "
                "cross-pool leakage.",
            )
            require(
                not (data_meta.get("split_migration") or {}).get(
                    "test_used_as_validation", False
                ),
                "data.source_video_identity.namespaces requires an independent "
                "test split; the current config aliases test_index as "
                "validation (data.split_migration.test_used_as_validation).",
            )

    # early_stopping.mode: min with a selection monitor is a silent behavior
    # reversal: the selection rank key is always bigger-is-better (in
    # constrained mode it is (global_positive_recall, worst_game_positive_recall,
    # -negative_score_p999)), so mode: min would minimize something that the
    # contract maximizes, making it a misconfiguration rather than a deliberate
    # choice. Minimizing a selection score is meaningless and the default is
    # max, so rejecting it cannot break any legitimate config.
    _SELECTION_MONITOR_VALUES = frozenset({"selection_score", "selection"})
    early_cfg = config.get("early_stopping") or {}
    es_monitor = early_cfg.get("monitor", "selection_score")
    es_mode = early_cfg.get("mode", "max")
    if es_monitor in _SELECTION_MONITOR_VALUES and es_mode == "min":
        require(
            False,
            f"early_stopping.mode: min is meaningless with "
            f"early_stopping.monitor: {es_monitor!r}. The selection rank "
            "key is always bigger-is-better (in constrained mode that is "
            "(global_positive_recall, worst_game_positive_recall, "
            "-negative_score_p999)), so mode: min silently reverses the "
            "ordering. Set early_stopping.mode: max, or switch to a "
            "numeric monitor such as cross_entropy.",
        )

    if problems:
        raise ConfigSchemaError(problems)


def finalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a merged configuration.

    Returns a deep copy so callers' dicts are never mutated in place.
    Idempotent: finalizing an already-finalized config is a no-op.
    """
    finalized = copy.deepcopy(config)
    _apply_defaults(finalized)
    migrate_split_roles(finalized)
    check_removed_keys(finalized)
    validate_config(finalized)
    resolve_decision_threshold(finalized)
    semantic_validate(finalized)
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
            nested = _nested_key_schema(dotted)
            if nested is not None:
                walk(nested, dotted)

    walk(SCHEMA, "")
    return rows
