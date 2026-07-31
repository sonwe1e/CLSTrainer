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


def factory_path(kind: str, name: str) -> str:
    key = f"{kind}/{name}"
    return _NAMES.get(key, "")


def registered_names(kind: str) -> list[str]:
    return sorted(k.split("/", 1)[1] for k in _RESOLVERS if k.startswith(f"{kind}/"))
