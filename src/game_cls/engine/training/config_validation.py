from __future__ import annotations

from pathlib import Path
from typing import Any

from game_cls.data.image_spec import ImageSpec


def has_independent_test(config: dict) -> bool:
    """True when test is a real holdout, not validation in disguise."""
    migration = (config.get("data") or {}).get("split_migration") or {}
    return not bool(migration.get("test_used_as_validation", False))


def _dataloader_option(
    config: dict,
    role: str,
    name: str,
    default: Any,
) -> Any:
    root = config["dataloader"]
    scoped = root.get(role, {})
    if not isinstance(scoped, dict):
        raise TypeError(f"dataloader.{role} must be a mapping")
    return scoped.get(name, root.get(name, default))


def _validate_dataloader_config(config: dict) -> None:
    accelerator = str(config["device"]["accelerator"])
    allowed_contexts = {"spawn", "fork", "forkserver"}

    for role in ("train", "eval"):
        workers = int(_dataloader_option(config, role, "num_workers", 0))
        if workers < 0:
            raise ValueError(f"dataloader.{role}.num_workers must be non-negative")
        if workers == 0:
            continue

        context = _dataloader_option(
            config,
            role,
            "multiprocessing_context",
            "spawn" if accelerator == "npu" else None,
        )
        if context is not None:
            context = str(context)
        if context is not None and context not in allowed_contexts:
            raise ValueError(
                "dataloader multiprocessing_context must be one of "
                f"{sorted(allowed_contexts)}, got {context!r}"
            )
        if accelerator == "npu" and context != "spawn":
            raise RuntimeError(f"NPU dataloader.{role} must use spawn, got {context!r}")
        if float(_dataloader_option(config, role, "timeout_seconds", 180)) <= 0:
            raise ValueError(f"dataloader.{role}.timeout_seconds must be positive")
        if int(_dataloader_option(config, role, "prefetch_factor", 2)) <= 0:
            raise ValueError(f"dataloader.{role}.prefetch_factor must be positive")
        if int(_dataloader_option(config, role, "worker_num_threads", 1)) <= 0:
            raise ValueError(f"dataloader.{role}.worker_num_threads must be positive")


def validate_training_config(config: dict) -> None:
    _validate_dataloader_config(config)
    data_cfg = config["data"]
    ImageSpec.from_config(data_cfg)
    model_cfg = config["model"]
    evaluation_cfg = config["evaluation"]
    evaluation_amp_dtype = str(
        evaluation_cfg.get("amp_dtype", config["device"].get("amp_dtype", "bfloat16"))
    )
    if evaluation_amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("evaluation.amp_dtype must be float16 or bfloat16")
    if int(evaluation_cfg.get("parquet_row_group_size", 4096)) <= 0:
        raise ValueError("evaluation.parquet_row_group_size must be positive")
    selection_metric = evaluation_cfg.get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    supported_selection_metrics = {
        "global_f1_at_decision_threshold",
        "macro_game_f1_at_decision_threshold",
        "worst_game_f1_at_decision_threshold",
        "global_f1_tau099",
        "macro_game_f1_tau099",
        "worst_game_f1_tau099",
        "composite",
    }
    if selection_metric not in supported_selection_metrics:
        raise ValueError(f"Unsupported evaluation.selection_metric: {selection_metric}")
    if (
        selection_metric == "composite"
        and sum(
            float(value)
            for value in evaluation_cfg.get("selection_weights", {}).values()
        )
        <= 0
    ):
        raise ValueError(
            "Composite model selection requires positive selection weights"
        )
    smoke_mode = bool((config.get("experiment") or {}).get("smoke_mode", False))
    if not data_cfg.get("synthetic", False) and not smoke_mode:
        # Production acceptance gate: with every full-validation source
        # disabled there is no model selection at all; such a run would
        # silently train blind. Smoke tests (experiment.smoke_mode=true) probe
        # forward/spawn/augmentation/eval paths with full validation disabled
        # on purpose, so the production policy is explicitly lifted for them
        # (audit PR-E: test-env policy and production safety policy must not
        # fight each other).
        full_every = int(evaluation_cfg.get("val_full_every_steps", 0))
        quick_every = int(evaluation_cfg.get("val_quick_every_steps", 0))
        full_at_end = bool(evaluation_cfg.get("val_full_at_end", True))
        if full_every <= 0 and not full_at_end:
            raise RuntimeError(
                "Production training requires at least one full-validation "
                "source: set evaluation.val_full_every_steps > 0 or "
                "evaluation.val_full_at_end=true (best-checkpoint selection "
                "and early stopping need validation signals)."
            )
        if full_every <= 0 and quick_every <= 0 and not full_at_end:
            raise RuntimeError(
                "Production training has every evaluation disabled; "
                "enable at least a quick or full validation cadence."
            )
        factory = str(model_cfg.get("factory", ""))
        checkpoint_path = model_cfg.get("checkpoint_path")
        if not factory or factory.endswith(":build_demo_model"):
            raise RuntimeError(
                "Production training cannot use build_demo_model; "
                "set model.factory to the real model factory."
            )
        if "your_package" in factory or "REPLACE_ME" in factory:
            raise RuntimeError("Production model.factory is still a placeholder.")
        if not checkpoint_path:
            raise RuntimeError("Production training requires model.checkpoint_path.")
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Production checkpoint does not exist: {checkpoint_path}"
            )
        if data_cfg.get("backend", "png") == "packed_uint8":
            if int(data_cfg.get("packed_max_open_shards", 16)) <= 0:
                raise ValueError("data.packed_max_open_shards must be positive")
            # Audit P0-6 / PR-C: the DataLoader genuinely requires the val
            # packed index (PackedUint8Backend is built for train/val/test
            # alike), so config validate must demand it to the same standard as
            # train/test -- otherwise validation passes and the loader blows up.
            for key in ("train_packed_index", "val_packed_index", "test_packed_index"):
                packed_index = data_cfg.get(key)
                if not packed_index or not Path(packed_index).is_file():
                    raise FileNotFoundError(
                        f"Packed backend requires existing data.{key}: {packed_index}"
                    )
            for split in ("train", "val", "test"):
                packed_index = data_cfg.get(f"{split}_packed_index")
                if not packed_index:
                    continue
                packed_video_index = data_cfg.get(f"{split}_packed_video_index") or str(
                    Path(packed_index).with_name("packed_video_entries.parquet")
                )
                if not Path(packed_video_index).is_file():
                    raise FileNotFoundError(
                        "Packed backend requires the integer video index: "
                        f"{packed_video_index}"
                    )
    if data_cfg.get("require_independent_test", False) and not has_independent_test(
        config
    ):
        raise RuntimeError(
            "data.require_independent_test is enabled but no independent "
            "test split exists: add data.val_index/data.val_video_index so "
            "the test split is held out for a single final evaluation."
        )
    if config.get("distributed", {}).get("enabled", False) and not model_cfg.get(
        "freeze_cls_batchnorm_stats", True
    ):
        raise RuntimeError(
            "Distributed training with trainable cls BatchNorm statistics "
            "requires SyncBatchNorm; keep freeze_cls_batchnorm_stats=true."
        )
