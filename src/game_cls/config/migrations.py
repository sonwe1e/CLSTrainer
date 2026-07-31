from __future__ import annotations

import copy
from typing import Any

from . import loader as _loader


def _migrate_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a V1 config (identified by the absence of config_version) to V2.

    V1 is the current production schema (experiment/device/data/pair/sampler/
    model/loss/optimizer/...). We map it onto the V2 component-selector layout
    without renaming the original flat keys, so that a migrated config is a
    superset: the original flat keys remain for the legacy loader path while the
    new component selectors drive the extensible path.
    """
    config = copy.deepcopy(raw)

    task = config.setdefault("task", {})
    if not task:
        task["type"] = "dual_frame_binary"
        task["factory"] = "game_cls.tasks.dual_frame_binary:build_task"
        task["params"] = {
            "positive_class_index": int(config.get("model", {}).get("num_classes", 2)) - 1,
            "num_classes": int(config.get("model", {}).get("num_classes", 2)),
        }

    trainable = config.setdefault("trainable", {})
    if not trainable:
        token = str(config.get("model", {}).get("trainable_name_contains", "cls"))
        trainable["policy"] = {
            "type": "name_token",
            "factory": "game_cls.trainable.name_token:build_policy",
            "params": {
                "token": token,
                "case_sensitive": True,
                "freeze_trainable_batchnorm_stats": bool(
                    config.get("model", {}).get("freeze_cls_batchnorm_stats", True)
                ),
                "freeze_frozen_batchnorm_stats": bool(
                    config.get("model", {}).get("freeze_backbone_batchnorm_stats", True)
                ),
            },
        }

    data = config.setdefault("data", {})
    if "module_factory" not in data:
        data["module_factory"] = "game_cls.data.module:build_game_video_pair_data_module"
    if "index_codec" not in data:
        data["index_codec"] = {"type": "legacy_game_binary"}
    existing_backend = data.get("backend", "png")
    if not isinstance(existing_backend, dict):
        data["backend"] = {"type": str(existing_backend), "params": {}}

    sampler = config.setdefault("sampler", {})
    if "policy" not in sampler:
        sampler["policy"] = {
            "type": "balanced_game_label_delta",
            "params": {
                "game_alpha": float(config.get("sampler", {}).get("game_alpha", 0.25)),
                "class_probability": {
                    int(key): float(value)
                    for key, value in config.get("sampler", {}).get(
                        "class_probability", {0: 0.5, 1: 0.5}
                    ).items()
                },
                "deduplicate_within_global_batch": bool(
                    config.get("sampler", {}).get("deduplicate_within_global_batch", True)
                ),
            },
        }

    runtime = config.setdefault("runtime", {})
    if not runtime:
        runtime["accelerator"] = {
            "type": str(config.get("device", {}).get("accelerator", "cpu"))
        }
        distributed_cfg = config.get("distributed", {})
        if bool(distributed_cfg.get("enabled", False)):
            runtime["distributed"] = {
                "type": "ddp",
                "params": {
                    "backend": str(distributed_cfg.get("backend", "nccl")),
                    "find_unused_parameters": bool(
                        distributed_cfg.get("find_unused_parameters", False)
                    ),
                    "broadcast_buffers": bool(
                        distributed_cfg.get("broadcast_buffers", False)
                    ),
                    "gradient_as_bucket_view": bool(
                        distributed_cfg.get("gradient_as_bucket_view", True)
                    ),
                },
            }
        else:
            runtime["distributed"] = {"type": "single_process", "params": {}}

    evaluation = config.setdefault("evaluation", {})
    if "suite" not in evaluation:
        evaluation["suite"] = {"type": "binary_threshold"}
    if "decision" not in evaluation:
        evaluation["decision"] = {
            "type": "threshold",
            "params": {
                "threshold": float(config.get("evaluation", {}).get("threshold", 0.99))
            },
        }

    config["config_version"] = 2
    return config


def migrate_to_latest(raw: dict[str, Any]) -> dict[str, Any]:
    """Detect config version and migrate to the latest (V2)."""
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping at the top level.")
    version = raw.get("config_version")
    if version is None:
        migrated = _migrate_v1_to_v2(raw)
        print("[CONFIG MIGRATION] Loaded v1 config and normalized to v2.")
        return migrated
    if version == 2:
        return copy.deepcopy(raw)
    raise ValueError(f"Unsupported config_version: {version}")
