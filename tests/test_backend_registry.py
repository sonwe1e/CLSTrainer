from __future__ import annotations

import pytest

from game_cls.data.backends import (
    PngBackendFactory,
    build_backend,
    register_backend,
    resolve_backend_factory,
)
from game_cls.data.image_spec import ImageSpec


def test_png_backend_registered() -> None:
    factory = resolve_backend_factory("png")
    assert isinstance(factory, PngBackendFactory)


def test_build_png_backend_has_expected_capabilities() -> None:
    spec = ImageSpec(width=448, height=208, channels=3)
    backend = build_backend("png", {}, spec, "train")
    assert backend.backend_name == "png"
    assert backend.capabilities.random_access is True
    assert backend.capabilities.spawn_safe is True


def test_duplicate_backend_rejected() -> None:
    class _Factory:
        capabilities = None

        def create(self, config, image_spec, split):
            return None

    factory = _Factory()
    with pytest.raises(ValueError):
        register_backend("png")(factory)


def test_resolve_unknown_backend() -> None:
    with pytest.raises(KeyError, match="Unknown data backend"):
        resolve_backend_factory("nonexistent_backend")
