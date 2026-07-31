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


def _suggest_key(unknown: str, known: list[str]) -> str | None:
    """Return the closest known key (by prefix/levenshtein-ish heuristic) or None."""
    if not known:
        return None
    prefix_matches = [k for k in known if k.startswith(unknown.split(".")[-1][:3])]
    if prefix_matches:
        return min(prefix_matches, key=lambda k: abs(len(k) - len(unknown)))
    return min(known, key=lambda k: abs(len(k) - len(unknown)))


def apply_overrides(
    config: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    known_keys = _collect_dotted_keys(config)
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
        if dotted_key not in known_keys:
            suggestion = _suggest_key(dotted_key, sorted(known_keys))
            msg = f"Override key {dotted_key!r} is unknown."
            if suggestion:
                msg += f" Did you mean: {suggestion}"
            raise ValueError(msg)
    return result


def _collect_dotted_keys(config: dict[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in config.items():
        dotted = f"{prefix}.{key}" if prefix else key
        keys.add(dotted)
        if isinstance(value, dict):
            keys.update(_collect_dotted_keys(value, dotted))
    return keys


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file, resolving any ``base`` inheritance.

    This function does **not** apply overrides — callers that need overrides
    should use :func:`apply_overrides` separately (typically *after* V1→V2
    migration so that users can override the new component-selector fields).
    """
    config_path = Path(path).resolve()
    raw = _yaml().safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_name = raw.pop("base", None)
    if base_name:
        base = load_config(config_path.parent / base_name)
        raw = deep_merge(base, raw)
    return raw
