from __future__ import annotations

from typing import Any

from ...contracts.data import BackendCapabilities, BackendSpec, DataBackendFactory, FrameBackend


class FrameBackendFactory(DataBackendFactory):
    """Base class for backend factories.

    A factory knows how to validate its config model and build a backend from
    a :class:`BackendSpec` (USERPLAN §12.2). Concrete factories register
    themselves via :func:`register_backend`.
    """

    capabilities: BackendCapabilities

    @property
    def config_model(self) -> type:
        return dict

    def create(self, config: Any, image_spec: Any, split: str) -> FrameBackend:  # type: ignore[override]
        raise NotImplementedError

    def from_spec(self, spec: BackendSpec, image_spec: Any) -> FrameBackend:
        return self.create(spec.params, image_spec, spec.split)


class PngDecoderMixin:
    """Shared PNG decode logic used by the PNG backend."""

    def _decode(self, reference: Any) -> Any:
        try:
            from PIL import Image
            from torchvision.transforms.v2 import functional as F
        except ImportError as exc:
            raise RuntimeError("Decoding pairs requires torch, torchvision and Pillow") from exc
        with Image.open(str(reference)) as image:
            return F.to_image(image.convert("RGB"))
