from __future__ import annotations

from typing import Any

from ...contracts.data import BackendCapabilities, FrameBackend
from .base import FrameBackendFactory
from .registry import register_backend


class PackedUint8Backend(FrameBackend):
    """Adapter over the existing :class:`game_cls.data.packed_backend.PackedUint8Backend`.

    The legacy implementation already exposes ``get_many``/``close``/``__call__`` and
    supports worker-lazy serialization via ``__getstate__``. We wrap it behind the
    :class:`FrameBackend` facade so the rest of the pipeline never branches on the
    backend name (USERPLAN §12.2).
    """

    backend_name = "packed_uint8"
    capabilities = BackendCapabilities(
        batch_decode=True,
        random_access=True,
        supports_preview=True,
        spawn_safe=True,
    )

    def __init__(self, legacy_backend: Any) -> None:
        self._backend = legacy_backend

    def get(self, reference: Any) -> Any:
        return self._backend(reference)

    def get_many(self, references: Any) -> Any:
        return self._backend.get_many(references)

    def preview(self, reference: Any, output_path: Any) -> None:
        import numpy as np
        from PIL import Image

        tensor = self._backend(reference)
        array = tensor.detach().cpu().permute(1, 2, 0).numpy()
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(str(output_path))

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "PackedUint8Backend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __getstate__(self) -> dict:
        return self._backend.__getstate__()

    def __setstate__(self, state: dict) -> None:
        # Rebuild from the serializable spec on the worker side.
        from game_cls.data.packed_backend import PackedUint8Backend as _Legacy

        self._backend = _Legacy.__new__(_Legacy)
        self._backend.__setstate__(state)


class PackedUint8BackendFactory(FrameBackendFactory):
    """Builds a :class:`PackedUint8Backend` from the packed-index params."""

    def create(self, config: Any, image_spec: Any, split: str) -> FrameBackend:
        from game_cls.data.packed_backend import PackedUint8Backend as _Legacy

        index_path = config["index_path"]
        max_open_shards = int(config.get("max_open_shards", 16))
        legacy = _Legacy(index_path, image_spec=image_spec, max_open_shards=max_open_shards)
        return PackedUint8Backend(legacy)


register_backend("packed_uint8")(PackedUint8BackendFactory())
