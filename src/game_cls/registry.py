from __future__ import annotations

from typing import Any, Callable

_RESOLVERS: dict[str, Callable[..., Any]] = {}
_NAMES: dict[str, str] = {}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a factory under ``kind``/``name``.

    Usage::

        @register("task", "dual_frame_binary")
        def build_task(config, image_spec):
            ...
    """

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        key = f"{kind}/{name}"
        if key in _RESOLVERS:
            raise ValueError(f"Duplicate registration for {key}")
        _RESOLVERS[key] = factory
        _NAMES[key] = factory.__module__ + ":" + factory.__qualname__
        return factory

    return decorator


def resolve(kind: str, name: str) -> Callable[..., Any]:
    key = f"{kind}/{name}"
    if key not in _RESOLVERS:
        available = sorted(k for k in _RESOLVERS if k.startswith(f"{kind}/"))
        raise KeyError(
            f"Unknown {kind}: {name!r}. Registered {kind}s: {available}"
        )
    return _RESOLVERS[key]


def import_from_path(factory_path: str) -> Callable[..., Any]:
    """Import a factory callable from a dotted path (``package.module:callable``)."""
    if ":" in factory_path:
        module_path, qualname = factory_path.rsplit(":", 1)
    elif "." in factory_path:
        module_path, qualname = factory_path.rsplit(".", 1)
    else:
        raise ValueError(
            f"factory_path must be 'module:callable' or 'module.callable', "
            f"got {factory_path!r}"
        )
    import importlib

    module = importlib.import_module(module_path)
    try:
        return getattr(module, qualname)
    except AttributeError as exc:
        raise ImportError(
            f"Cannot import {qualname!r} from {module_path}: {exc}"
        ) from exc


def resolve_component(
    kind: str,
    type_name: str,
    factory_path: str = "",
) -> Callable[..., Any]:
    """Resolve a component factory, supporting both registry and factory_path.

    When ``factory_path`` is provided, it takes precedence and is dynamically
    imported. This lets users plug in custom components without modifying the
    registry (USERPLAN §12 factory-path support).
    """
    if factory_path:
        return import_from_path(factory_path)
    return resolve(kind, type_name)


def factory_path(kind: str, name: str) -> str:
    key = f"{kind}/{name}"
    return _NAMES.get(key, "")


def registered_names(kind: str) -> list[str]:
    return sorted(k.split("/", 1)[1] for k in _RESOLVERS if k.startswith(f"{kind}/"))
