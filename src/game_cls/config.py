from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from game_cls.config_schema import (
    check_override_path,
    finalize_config,
)


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
    """Apply ``key=value`` overrides.

    Overrides may only address keys known to the configuration schema.
    Typos such as ``optimzier.learning_rate`` raise immediately instead of
    silently creating a field that nobody reads.
    """
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        dotted_key, raw_value = item.split("=", 1)
        check_override_path(dotted_key)
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
    """Load, merge and validate a configuration file.

    The returned config is schema-validated: unknown keys, removed keys and
    type mismatches all raise ``ConfigSchemaError``. ``decision.threshold``
    is propagated to the loss and evaluation sections so the business
    threshold has exactly one source of truth.
    """
    raw, _ = _load_raw_with_sources(Path(path).resolve())
    merged = apply_overrides(raw, overrides or [])
    return finalize_config(merged)


def _load_raw_with_sources(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        # JSON configs (e.g. a run's resolved_config.json) must use the JSON
        # parser: PyYAML's 1.1 resolver treats numbers like 1e-05 as strings.
        raw = json.loads(text)
    else:
        raw = _yaml().safe_load(text) or {}
    sources: dict[str, str] = {}
    base_name = raw.pop("base", None)
    if base_name:
        base, sources = _load_raw_with_sources(config_path.parent / base_name)
    else:
        base = {}
    origin = str(config_path)
    _record_sources(sources, raw, origin)
    return deep_merge(base, raw), sources


def _record_sources(
    sources: dict[str, str], node: dict[str, Any], origin: str, prefix: str = ""
) -> None:
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _record_sources(sources, value, origin, dotted)
        else:
            sources[dotted] = origin


def load_config_with_sources(
    path: str | Path, overrides: list[str] | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Like ``load_config`` but also reports where each value came from.

    The sources map covers the RAW merge tree (base files and this file);
    command line overrides are recorded as ``override:<item>``.
    """
    raw, sources = _load_raw_with_sources(Path(path).resolve())
    for item in overrides or []:
        if "=" in item:
            dotted_key = item.split("=", 1)[0]
            sources[dotted_key] = f"override:{item}"
    merged = apply_overrides(raw, overrides or [])
    return finalize_config(merged), sources
