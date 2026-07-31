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
        self._backends: list[Any] = []

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
        """Build dataloaders using the legacy pipeline.

        Delegates to :func:`build_legacy_loader_bundle` in
        ``data.legacy_pipeline`` — the single source of truth for the default
        data loading behavior. This avoids the circular import that would arise
        if we imported ``_make_dataloaders`` from ``engine.trainer`` directly.
        """
        from .legacy_pipeline import build_legacy_loader_bundle

        # Close any previously tracked backends (e.g. on rebuild) before
        # clearing references. Using close() ensures memmaps and file
        # descriptors are released rather than leaked.
        self.close()
        bundle = build_legacy_loader_bundle(
            self._config, self._image_spec, runtime
        )
        # Track backends for cleanup. The datasets hold references to the
        # backends; we collect them here so we can close memmaps/file
        # descriptors when training finishes.
        # NOTE: in the synthetic path, eval DataLoaders wrap their dataset in a
        # torch.utils.data.Subset, so we unwrap Subset (if present) to reach
        # the underlying dataset's decoder. In the production path the datasets
        # are not wrapped, so the unwrap is a no-op.
        for loader in (bundle.train, bundle.quick_test, bundle.full_test):
            dataset = getattr(loader, "dataset", None)
            if dataset is None:
                continue
            # Unwrap Subset to reach the underlying dataset.
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            decoder = getattr(dataset, "decoder", None)
            if decoder is not None and decoder not in self._backends:
                self._backends.append(decoder)
        return bundle

    def close(self) -> None:
        """Close all tracked backends (memmaps, file descriptors)."""
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                pass
        self._backends.clear()


def build_game_video_pair_data_module(config: Any, image_spec: Any) -> LegacyGameVideoDataModule:
    """Factory function referenced by the V2 config migration.

    The migration writes ``data.module_factory =
    game_cls.data.module:build_game_video_pair_data_module`` so that the
    DataModule can be swapped without modifying the builder. This factory
    returns the default :class:`LegacyGameVideoDataModule` that wraps the
    existing legacy data pipeline.
    """
    return LegacyGameVideoDataModule(config, image_spec)
