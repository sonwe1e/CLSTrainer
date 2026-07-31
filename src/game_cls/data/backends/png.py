from __future__ import annotations

from typing import Any

from ...contracts.data import BackendCapabilities, FrameBackend
from .base import FrameBackendFactory, PngDecoderMixin
from .registry import register_backend


class PngBackend(FrameBackend):
    """Decode individual PNG files by path (the original decode path)."""

    backend_name = "png"
    capabilities = BackendCapabilities(
        batch_decode=False,
        random_access=True,
        supports_preview=True,
        spawn_safe=True,
    )

    def __init__(self, image_spec: Any, decoder: Any) -> None:
        self.image_spec = image_spec
        self._decoder = decoder

    def get(self, reference: Any) -> Any:
        return self._decoder(reference)

    def get_many(self, references: Any) -> Any:
        import torch

        decoded = [self._decoder(ref) for ref in references]
        return torch.stack(decoded)

    def preview(self, reference: Any, output_path: Any) -> None:
        from PIL import Image

        with Image.open(str(reference)) as image:
            image.convert("RGB").save(str(output_path))

    def close(self) -> None:
        pass


class PngBackendFactory(FrameBackendFactory, PngDecoderMixin):
    """Builds a :class:`PngBackend`. PNG needs no extra params."""

    def create(self, config: Any, image_spec: Any, split: str) -> FrameBackend:
        del config, split
        return PngBackend(image_spec=image_spec, decoder=self._decode)

    @classmethod
    def create_from_legacy_data_config(
        cls, data_config: Any, split: str, image_spec: Any
    ) -> FrameBackend:
        """Build a PNG backend from the legacy flat ``data`` config section."""
        del data_config, split
        return PngBackend(image_spec=image_spec, decoder=cls()._decode)


register_backend("png")(PngBackendFactory())
