from __future__ import annotations

from typing import Any

from ...contracts.data import FrameBackend
from .base import FrameBackendFactory

_FACTORIES: dict[str, FrameBackendFactory] = {}


def register_backend(name: str) -> Any:
    """Decorator registering a backend factory under ``name``."""

    def decorator(factory: FrameBackendFactory) -> FrameBackendFactory:
        if name in _FACTORIES:
            raise ValueError(f"Duplicate backend registration: {name}")
        _FACTORIES[name] = factory
        return factory

    return decorator


def resolve_backend_factory(name: str) -> FrameBackendFactory:
    if name not in _FACTORIES:
        raise KeyError(f"Unknown data backend: {name!r}. Registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name]


def build_backend(name: str, config: Any, image_spec: Any, split: str) -> FrameBackend:
    return resolve_backend_factory(name).create(config, image_spec, split)


def registered_backends() -> list[str]:
    return sorted(_FACTORIES)
