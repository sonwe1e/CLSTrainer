from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ImageSpec:
    """Configured image layout shared by indexing, packing and training."""

    width: int
    height: int
    channels: int

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> "ImageSpec":
        spec = cls(
            width=int(data_config["width"]),
            height=int(data_config["height"]),
            channels=int(data_config.get("channels", 3)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.width <= 0:
            raise ValueError(
                f"Image width must be positive, got {self.width}"
            )
        if self.height <= 0:
            raise ValueError(
                f"Image height must be positive, got {self.height}"
            )
        if self.channels <= 0:
            raise ValueError(
                f"Image channels must be positive, got {self.channels}"
            )

    @property
    def chw(self) -> tuple[int, int, int]:
        return self.channels, self.height, self.width

    @property
    def hw(self) -> tuple[int, int]:
        return self.height, self.width

    def validate_pair_batch_shape(self, shape: Sequence[int]) -> None:
        actual = tuple(int(item) for item in shape)
        expected_tail = (2, *self.chw)
        if len(actual) != 5 or actual[1:] != expected_tail:
            raise RuntimeError(
                "Training image shape mismatch: "
                f"expected [B,2,{self.channels},{self.height},{self.width}], "
                f"got {actual}"
            )
