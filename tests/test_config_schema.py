from __future__ import annotations

import pytest

from game_cls.config import load_config
from game_cls.config.migrations import migrate_to_latest
from game_cls.config.schema import validate_and_normalize_config, ValidationError


def _v2_config() -> dict:
    return {
        "config_version": 2,
        "task": {"type": "dual_frame_binary", "factory": "game_cls.tasks.dual_frame_binary:build_task", "params": {"positive_class_index": 1, "num_classes": 2}},
        "trainable": {"policy": {"type": "name_token", "factory": "game_cls.trainable.name_token:build_policy", "params": {"token": "cls"}}},
        "data": {"index_codec": {"type": "legacy_game_binary"}, "backend": {"type": "png", "params": {}}},
        "sampler": {"policy": {"type": "balanced_game_label_delta", "params": {"game_alpha": 0.25, "class_probability": {"0": 0.5, "1": 0.5}}}},
        "runtime": {"accelerator": {"type": "cpu"}, "distributed": {"type": "single_process", "params": {}}},
        "evaluation": {"suite": {"type": "binary_threshold"}, "decision": {"type": "threshold", "params": {"threshold": 0.99}}},
    }


def test_v2_config_validates() -> None:
    cfg = validate_and_normalize_config(_v2_config())
    assert cfg.task.type == "dual_frame_binary"
    assert cfg.trainable.policy.type == "name_token"
    assert cfg.runtime.accelerator.type == "cpu"


def test_unknown_selector_key_rejected() -> None:
    bad = _v2_config()
    bad["task"]["bogus"] = True
    with pytest.raises(ValidationError):
        validate_and_normalize_config(bad)


def test_unknown_key_inside_params_tolerated_at_top_level() -> None:
    """Plugin ``params`` are open dicts at the top level; per-plugin validation
    happens at build time (USERPLAN §8.2). The top-level schema only enforces
    the selector keys (type/factory/params)."""
    config = _v2_config()
    config["evaluation"]["decision"]["params"]["threshhold"] = 0.5  # typo, but inside params
    cfg = validate_and_normalize_config(config)
    assert cfg.evaluation.decision.type == "threshold"


def test_legacy_top_level_key_tolerated() -> None:
    config = _v2_config()
    config["experiment"] = {"name": "x", "seed": 1}
    config["model"] = {"factory": "x:y:z"}
    cfg = validate_and_normalize_config(config)
    assert cfg.task.type == "dual_frame_binary"


def test_v1_migrates_and_validates() -> None:
    raw = load_config("configs/npu_1p.yaml")
    migrated = migrate_to_latest(raw)
    assert migrated["config_version"] == 2
    cfg = validate_and_normalize_config(migrated)
    assert cfg.runtime.accelerator.type == "npu"
    assert cfg.trainable.policy.type == "name_token"
    assert cfg.sampler.policy.type == "balanced_game_label_delta"


def test_missing_required_field_gives_friendly_error() -> None:
    bad = _v2_config()
    bad["task"] = {"factory": "x:y:z"}
    with pytest.raises(ValidationError, match="validation failed"):
        validate_and_normalize_config(bad)
