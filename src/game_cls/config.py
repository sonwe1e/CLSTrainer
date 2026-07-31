from __future__ import annotations

from typing as _typing

from .config import loader as _loader
from .config import migrations as _migrations
from .config import schema as _schema


def deep_merge(base: dict, override: dict) -> dict:
    return _loader.deep_merge(base, override)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    return _loader.apply_overrides(config, overrides)


def load_config(path, overrides=None) -> dict:
    return _loader.load_config(path, overrides)


def load_and_validate_config(path, overrides=None) -> _schema.ExperimentConfig:
    """Load, migrate, and validate a config into the typed V2 schema."""
    raw = _loader.load_config(path, overrides)
    migrated = _migrations.migrate_to_latest(raw)
    return _schema.validate_and_normalize_config(migrated)


__all__ = [
    "load_config",
    "apply_overrides",
    "deep_merge",
    "load_and_validate_config",
]
