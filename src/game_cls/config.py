from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from game_cls.config_schema import (
    ConfigSchemaError,
    check_override_path,
    finalize_config,
)

# Preset groups a recipe may reference, mapped to their configs subdirectory.
PRESET_GROUPS = ("augmentation", "dataloader", "evaluation")

RECIPE_MARKER_KEYS = ("profile", "presets", "task_profile")


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


def _read_raw_file(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        # JSON configs (e.g. a run's resolved_config.json) must use the JSON
        # parser: PyYAML's 1.1 resolver treats numbers like 1e-05 as strings.
        return json.loads(text)
    return _yaml().safe_load(text) or {}


RECIPE_LEVEL_KEYS = ("profile", "task_profile", "presets")


def _split_overrides(
    raw: dict[str, Any], overrides: list[str]
) -> tuple[list[str], list[str]]:
    """Split overrides into recipe-level and config-level groups.

    Recipe-level targets (``profile``, ``task_profile``, ``presets.<group>``)
    are honored only when the entry file actually is a recipe.
    """
    if not _is_recipe(raw):
        return [], list(overrides)
    recipe_overrides: list[str] = []
    config_overrides: list[str] = []
    for item in overrides:
        dotted = item.split("=", 1)[0] if "=" in item else item
        head = dotted.split(".")[0]
        if head in RECIPE_LEVEL_KEYS:
            recipe_overrides.append(item)
        else:
            config_overrides.append(item)
    return recipe_overrides, config_overrides


def _apply_recipe_overrides(
    raw: dict[str, Any], overrides: list[str], recipe_path: Path
) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        dotted_key, raw_value = item.split("=", 1)
        value = _parse_scalar(raw_value)
        parts = dotted_key.split(".")
        if parts[0] in ("profile", "task_profile"):
            if len(parts) != 1:
                raise ConfigSchemaError(
                    [f"Override target {dotted_key} is not a scalar key."]
                )
            raw[parts[0]] = value
            continue
        if parts[0] == "presets":
            if len(parts) != 2 or parts[1] not in PRESET_GROUPS:
                raise ConfigSchemaError(
                    [
                        f"Override target {dotted_key}: presets overrides must "
                        f"address one of {['presets.' + g for g in PRESET_GROUPS]}."
                    ]
                )
            presets = raw.setdefault("presets", {})
            if not isinstance(presets, dict):
                raise ConfigSchemaError(
                    [f"{recipe_path}: 'presets' must be a mapping."]
                )
            presets[parts[1]] = value
            continue
        raise ConfigSchemaError([f"Override target {dotted_key} is not recipe-level."])
    return raw


def _clear_legacy_thresholds_if_overridden(
    merged: dict[str, Any], overrides: list[str]
) -> None:
    """An explicit decision.threshold override wins over injected copies.

    Resolved configs (resolved_config.json) carry loss.threshold and
    evaluation.threshold mirrors of the old decision value; overriding
    decision.threshold must not trip the conflict guard against those
    stale mirrors.
    """
    override_keys = {item.split("=", 1)[0] for item in overrides if "=" in item}
    if "decision.threshold" not in override_keys:
        return
    for section_name in ("loss", "evaluation"):
        section = merged.get(section_name)
        if isinstance(section, dict):
            section.pop("threshold", None)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load, merge and validate a configuration file.

    A config may be either:

    * A recipe: ``profile`` + optional ``task_profile``/``presets`` plus only the
      fields the user actually decides. Merge order is
      task_profile -> profile -> presets -> recipe -> CLI overrides.
    * A resolved config (JSON, e.g. a run's ``resolved_config.json``): already
      flattened, loaded as-is.

    The returned config is schema-validated: unknown keys, removed keys and
    type mismatches all raise ``ConfigSchemaError``. ``decision.threshold``
    is propagated to the loss and evaluation sections so the business
    threshold has exactly one source of truth.
    """
    config_path = Path(path).resolve()
    overrides = list(overrides or [])
    peek = _read_raw_file(config_path)
    recipe_overrides, config_overrides = _split_overrides(peek, overrides)
    raw, _ = _load_raw_with_sources(config_path, recipe_overrides)
    merged = apply_overrides(raw, config_overrides)
    _clear_legacy_thresholds_if_overridden(merged, config_overrides)
    return finalize_config(merged)


def _resolve_layer_file(recipe_path: Path, subdir: str, name: str, kind: str) -> Path:
    """Locate ``<subdir>/<name>.yaml`` relative to the recipe's config tree."""
    if not str(name).strip():
        raise ConfigSchemaError([f"Empty {kind} name in {recipe_path}."])
    candidates: list[Path] = []
    tree_root = recipe_path.parent
    for _ in range(3):
        candidates.append(tree_root / subdir / f"{name}.yaml")
        tree_root = tree_root.parent
    candidates.append(Path("configs") / subdir / f"{name}.yaml")
    candidates.append(_configs_root() / subdir / f"{name}.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if "/" in subdir:
        group = subdir.split("/", 1)[1]
        available = [
            entry.split("/", 1)[1]
            for entry in list_available_layers().get("presets", [])
            if entry.startswith(group + "/")
        ]
    else:
        available = list_available_layers().get(subdir, [])
    available_text = f" Available: {', '.join(available)}." if available else ""
    raise ConfigSchemaError(
        [f"Unknown {kind} '{name}' referenced by {recipe_path}.{available_text}"]
    )


def _is_recipe(raw: dict[str, Any]) -> bool:
    return any(marker in raw for marker in RECIPE_MARKER_KEYS)


def _load_raw_with_sources(
    config_path: Path,
    recipe_overrides: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    raw = _read_raw_file(config_path)
    if _is_recipe(raw):
        if recipe_overrides:
            raw = _apply_recipe_overrides(raw, recipe_overrides, config_path)
        return _load_recipe_with_sources(config_path, raw)

    if "base" in raw:
        raise ConfigSchemaError(
            [
                f"{config_path}: the 'base:' inheritance mechanism was removed. "
                "Convert this file to a recipe that composes profile/presets "
                "(see configs/recipes/ for examples)."
            ]
        )
    # Non-recipe configs (e.g. a run's resolved_config.json) are already
    # flattened and load as-is.
    sources: dict[str, str] = {}
    origin = str(config_path)
    _record_sources(sources, raw, origin)
    return raw, sources


def _load_recipe_with_sources(
    recipe_path: Path, raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve task_profile/profile/presets and merge them under the recipe."""
    raw = copy.deepcopy(raw)
    if "contract" in raw:
        raise ConfigSchemaError(
            [
                f"{recipe_path}: the 'contract' layer has been renamed to "
                "'task_profile'. Use 'task_profile: dual_frame_binary' "
                "instead. Task profiles are defaults, not invariants — "
                "recipes may override every field."
            ]
        )
    contract_name = raw.pop("task_profile", "dual_frame_binary")
    profile_name = raw.pop("profile", None)
    presets = raw.pop("presets", {}) or {}
    if "base" in raw:
        raise ConfigSchemaError(
            [
                f"{recipe_path}: recipes use profile/presets, not 'base'. "
                "Remove the base key."
            ]
        )
    if not isinstance(presets, dict):
        raise ConfigSchemaError(
            [f"{recipe_path}: 'presets' must be a mapping of group: name."]
        )
    unknown_groups = sorted(set(presets) - set(PRESET_GROUPS))
    if unknown_groups:
        raise ConfigSchemaError(
            [
                f"{recipe_path}: unknown preset group(s) "
                f"{unknown_groups}. Allowed groups: {list(PRESET_GROUPS)}."
            ]
        )

    layers: list[Path] = []
    if contract_name:
        layers.append(
            _resolve_layer_file(
                recipe_path, "task_profiles", contract_name, "task_profile"
            )
        )
    if profile_name:
        layers.append(
            _resolve_layer_file(recipe_path, "profiles", profile_name, "profile")
        )
    for group in PRESET_GROUPS:
        if group in presets:
            layers.append(
                _resolve_layer_file(
                    recipe_path,
                    f"presets/{group}",
                    presets[group],
                    f"{group} preset",
                )
            )

    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for layer_path in layers:
        layer_raw = _read_raw_file(layer_path)
        if _is_recipe(layer_raw):
            raise ConfigSchemaError(
                [
                    f"{layer_path}: layer files (task_profile/profile/preset) must "
                    "be plain config sections, not recipes."
                ]
            )
        if "base" in layer_raw:
            raise ConfigSchemaError(
                [
                    f"{layer_path}: layer files cannot use 'base:'. Put the "
                    "inheritance in the recipe (profile/presets) instead."
                ]
            )
        _record_sources(sources, layer_raw, str(layer_path))
        merged = deep_merge(merged, layer_raw)
    _record_sources(sources, raw, str(recipe_path))
    merged = deep_merge(merged, raw)
    return merged, sources


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

    The sources map covers the full merge tree (task_profile/profile/presets,
    plus the entry file); command line overrides are recorded as
    ``override:<item>``.
    """
    config_path = Path(path).resolve()
    overrides = list(overrides or [])
    peek = _read_raw_file(config_path)
    recipe_overrides, config_overrides = _split_overrides(peek, overrides)
    raw, sources = _load_raw_with_sources(config_path, recipe_overrides)
    for item in overrides:
        if "=" in item:
            dotted_key = item.split("=", 1)[0]
            sources[dotted_key] = f"override:{item}"
    merged = apply_overrides(raw, config_overrides)
    _clear_legacy_thresholds_if_overridden(merged, config_overrides)
    return finalize_config(merged), sources


def _configs_root() -> Path:
    """The configs tree root: CWD ``./configs`` when present (a repo checkout),
    else the copy shipped inside the installed wheel (audit PR-F).

    data-files install under ``sys.prefix`` (setuptools does not place them in
    site-packages), so the installed tree may be either next to the package or
    at the environment root.
    """
    import sys

    cwd_root = Path("configs")
    if cwd_root.is_dir():
        return cwd_root
    for candidate in (
        Path(__file__).resolve().parent / "configs",
        Path(sys.prefix) / "game_cls" / "configs",
    ):
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent / "configs"


def list_available_layers() -> dict[str, list[str]]:
    """Available task profiles/profiles/presets under ./configs, for errors and init."""
    root = _configs_root()
    result: dict[str, list[str]] = {}
    for subdir in ("task_profiles", "profiles"):
        directory = root / subdir
        result[subdir] = (
            sorted(path.stem for path in directory.glob("*.yaml"))
            if directory.is_dir()
            else []
        )
    presets: list[str] = []
    presets_root = root / "presets"
    if presets_root.is_dir():
        for group_dir in sorted(presets_root.iterdir()):
            if group_dir.is_dir():
                for path in sorted(group_dir.glob("*.yaml")):
                    presets.append(f"{group_dir.name}/{path.stem}")
    result["presets"] = presets
    return result
