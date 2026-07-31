from __future__ import annotations

import pytest

from game_cls import registry


def test_register_and_resolve() -> None:
    @registry.register("demo", "alpha")
    def build_alpha(config):
        return "alpha-" + str(config)

    assert registry.resolve("demo", "alpha")(42) == "alpha-42"
    assert "alpha" in registry.registered_names("demo")


def test_duplicate_registration_rejected() -> None:
    @registry.register("demo", "beta")
    def build_beta(config):
        return config

    with pytest.raises(ValueError):
        @registry.register("demo", "beta")
        def build_beta2(config):
            return config


def test_resolve_unknown_reports_available() -> None:
    with pytest.raises(KeyError, match="Registered demo"):
        registry.resolve("demo", "nonexistent")
