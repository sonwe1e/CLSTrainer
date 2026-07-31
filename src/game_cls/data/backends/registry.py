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


def build_backend_from_legacy_data_config(
    data_config: Any, backend_selector: Any, split: str, image_spec: Any
) -> FrameBackend:
    """Build a backend from the legacy flat ``data`` config.

    During migration the ``data.backend`` selector carries no ``index_path``,
    so we route through each factory's ``create_from_legacy_data_config`` which
    knows how to find the path from the legacy ``data.{split}_packed_index``.
    """
    name = str(backend_selector.get("type", "png"))
    factory = resolve_backend_factory(name)
    if hasattr(factory, "create_from_legacy_data_config"):
        return factory.create_from_legacy_data_config(data_config, split, image_spec)
    # Factories without a legacy adapter fall back to the standard create.
    return factory.create(backend_selector.get("params", {}), image_spec, split)


def registered_backends() -> list[str]:
    return sorted(_FACTORIES)
