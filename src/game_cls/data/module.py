from __future__ import annotations

from typing import Any

from ..contracts.data import LoaderBundle


class LegacyGameVideoDataModule:
    """Default DataModule that wraps the existing legacy data pipeline.

    This is the first step of the DataModule migration (PLAN3 §九 D1): it
    moves the existing ``_make_dataloaders`` logic behind the
    :class:`DataModule` interface without changing any behavior. The trainer
    talks to this interface and never branches on backend name or sampling
    algorithm directly.
    """

    def __init__(self, config: Any, image_spec: Any) -> None:
        self._config = config
        self._image_spec = image_spec

    @property
    def config(self) -> Any:
        return self._config

    @property
    def image_spec(self) -> Any:
        return self._image_spec

    def build_loaders(
        self,
        *,
        runtime: Any,
        state_mode: str = "full",
    ) -> LoaderBundle:
        """Build dataloaders using the legacy pipeline."""
        from ..engine.trainer import _make_dataloaders

        rank = int(runtime.distributed.rank)
        world_size = int(runtime.distributed.world_size)
        return _make_dataloaders(self._config, rank, world_size)
