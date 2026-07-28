from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def _yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: python -m pip install PyYAML") from exc
    return yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(value: str) -> Any:
    return _yaml().safe_load(value)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override nested key below {key!r}")
            cursor = child
        cursor[keys[-1]] = _parse_scalar(raw_value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = _yaml().safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_name = raw.pop("base", None)
    if base_name:
        base = load_config(config_path.parent / base_name)
        raw = deep_merge(base, raw)
    return apply_overrides(raw, overrides or [])

