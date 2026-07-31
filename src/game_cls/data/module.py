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

        # Clear any previously tracked backends (e.g. on rebuild).
        self._backends.clear()
        bundle = build_legacy_loader_bundle(
            self._config, self._image_spec, runtime
        )
        # Track backends for cleanup. The datasets hold references to the
        # backends; we collect them here so we can close memmaps/file
        # descriptors when training finishes.
        # NOTE: eval DataLoaders wrap their dataset in a torch.utils.data.Subset,
        # so we must unwrap Subset to reach the underlying dataset's decoder.
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
