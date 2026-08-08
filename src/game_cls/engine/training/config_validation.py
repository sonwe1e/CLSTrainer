from __future__ import annotations

from pathlib import Path
from typing import Any

from game_cls.data.image_spec import ImageSpec


def has_independent_test(config: dict) -> bool:
    """True when validation and test point to distinct contract-5 inputs."""
    data = config.get("data") or {}
    if data.get("synthetic", False):
        return True
    val_index = data.get("val_index")
    test_index = data.get("test_index")
    return bool(val_index and test_index and test_index != val_index)


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
    if name in scoped:
        return scoped[name]
    if name in {"multiprocessing_context", "timeout_seconds", "worker_num_threads"}:
        return root.get(name, default)
    return default


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
    if not data_cfg.get("synthetic", False):
        backend = str(data_cfg.get("backend", "png"))
        video_key = (
            "val_packed_video_index" if backend == "packed_uint8" else "val_video_index"
        )
        required_validation = {
            "val_index": data_cfg.get("val_index"),
            video_key: data_cfg.get(video_key),
        }
        missing_validation = [
            key for key, value in required_validation.items() if not value
        ]
        if missing_validation:
            raise ValueError(
                "Real-data training requires dedicated validation inputs: "
                + ", ".join(f"data.{key}" for key in missing_validation)
            )
        if data_cfg.get("test_index") == data_cfg.get("val_index"):
            raise ValueError(
                "data.test_index must be omitted or distinct from data.val_index."
            )
        test_video_key = (
            "test_packed_video_index"
            if backend == "packed_uint8"
            else "test_video_index"
        )
        test_parts = (data_cfg.get("test_index"), data_cfg.get(test_video_key))
        if any(test_parts) and not all(test_parts):
            raise ValueError(
                "An independent test split requires both data.test_index and "
                f"data.{test_video_key}."
            )
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
            packed_keys = ["train_packed_index", "val_packed_index"]
            if has_independent_test(config):
                packed_keys.append("test_packed_index")
            for key in packed_keys:
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
            "test split exists: configure dedicated data.test_index and "
            "data.test_video_index inputs."
        )
    if config.get("distributed", {}).get("enabled", False) and not model_cfg.get(
        "freeze_cls_batchnorm_stats", True
    ):
        raise RuntimeError(
            "Distributed training with trainable cls BatchNorm statistics "
            "requires SyncBatchNorm; keep freeze_cls_batchnorm_stats=true."
        )
