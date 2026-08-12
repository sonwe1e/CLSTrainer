from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairedAugmentConfig:
    """Configuration for augmenting a dual-frame sample.

    Geometric transforms are always shared by both frames so that augmentation
    never invents artificial motion. Color parameters can also be shared (the
    default) to preserve the relative appearance between the two frames.
    """

    enabled: bool = False
    horizontal_flip_p: float = 0.0
    crop_scale: tuple[float, float] = (1.0, 1.0)
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    gamma: tuple[float, float] = (1.0, 1.0)
    color_shared: bool = True
    noise_std: float = 0.0
    erase_p: float = 0.0
    erase_scale: tuple[float, float] = (0.02, 0.10)


class PairedAugment:
    """Apply safe, synchronized augmentation to ``[2, C, H, W]`` uint8 frames.

    The input and output are uint8 tensors. This keeps augmentation isolated
    inside the dataset and preserves the trainer's existing normalization path.
    """

    def __init__(self, config: PairedAugmentConfig) -> None:
        self.config = config

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        if low == high:
            return float(low)
        return float(torch.empty((), dtype=torch.float32).uniform_(low, high).item())

    def _crop_resize(self, images: torch.Tensor) -> torch.Tensor:
        low, high = self.config.crop_scale
        scale = self._uniform(low, high)
        if scale >= 0.999999:
            return images

        _, _, height, width = images.shape
        crop_h = max(1, min(height, int(round(height * scale))))
        crop_w = max(1, min(width, int(round(width * scale))))
        max_top = height - crop_h
        max_left = width - crop_w
        top = int(torch.randint(max_top + 1, ()).item()) if max_top else 0
        left = int(torch.randint(max_left + 1, ()).item()) if max_left else 0
        cropped = images[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(
            cropped,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _adjust_color(
        image: torch.Tensor,
        *,
        brightness_factor: float,
        contrast_factor: float,
        saturation_factor: float,
        gamma: float,
    ) -> torch.Tensor:
        x = image
        if brightness_factor != 1.0:
            x = x * brightness_factor
        if contrast_factor != 1.0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * contrast_factor + mean
        if saturation_factor != 1.0 and x.shape[0] == 3:
            # ITU-R BT.601 luma coefficients. Keep this local to avoid a
            # torchvision dependency in the Lite package.
            weights = x.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            gray = (x * weights).sum(dim=0, keepdim=True)
            x = (x - gray) * saturation_factor + gray
        x = x.clamp_(0.0, 1.0)
        if gamma != 1.0:
            x = x.clamp_min(1e-6).pow(gamma)
        return x.clamp_(0.0, 1.0)

    def _sample_color_params(self) -> tuple[float, float, float, float]:
        cfg = self.config
        brightness = self._uniform(1.0 - cfg.brightness, 1.0 + cfg.brightness)
        contrast = self._uniform(1.0 - cfg.contrast, 1.0 + cfg.contrast)
        saturation = self._uniform(1.0 - cfg.saturation, 1.0 + cfg.saturation)
        gamma = self._uniform(cfg.gamma[0], cfg.gamma[1])
        return brightness, contrast, saturation, gamma

    def _erase(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.erase_p <= 0.0 or float(torch.rand(()).item()) >= cfg.erase_p:
            return images

        _, _, height, width = images.shape
        scale = self._uniform(cfg.erase_scale[0], cfg.erase_scale[1])
        erase_h = max(1, min(height, int(round(height * scale**0.5))))
        erase_w = max(1, min(width, int(round(width * scale**0.5))))
        max_top = height - erase_h
        max_left = width - erase_w
        top = int(torch.randint(max_top + 1, ()).item()) if max_top else 0
        left = int(torch.randint(max_left + 1, ()).item()) if max_left else 0

        # Same erase rectangle for both frames; otherwise augmentation can create
        # a fake inter-frame object disappearance.
        images[:, :, top : top + erase_h, left : left + erase_w] = 0.0
        return images

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return images
        if images.ndim != 4 or images.shape[0] != 2:
            raise ValueError(
                f"PairedAugment expects [2,C,H,W], got {tuple(images.shape)}"
            )

        x = images.float().div_(255.0)
        x = self._crop_resize(x)

        if self.config.horizontal_flip_p > 0.0:
            if float(torch.rand(()).item()) < self.config.horizontal_flip_p:
                x = torch.flip(x, dims=(-1,))

        if self.config.color_shared:
            params = self._sample_color_params()
            x = torch.stack(
                [self._adjust_color(frame, brightness_factor=params[0], contrast_factor=params[1], saturation_factor=params[2], gamma=params[3]) for frame in x],
                dim=0,
            )
        else:
            x = torch.stack(
                [
                    self._adjust_color(
                        frame,
                        brightness_factor=params[0],
                        contrast_factor=params[1],
                        saturation_factor=params[2],
                        gamma=params[3],
                    )
                    for frame in x
                    for params in [self._sample_color_params()]
                ],
                dim=0,
            )

        if self.config.noise_std > 0.0:
            # Independent pixel noise is safe: geometry and color response stay
            # paired while small sensor/compression-like perturbations differ.
            x = x + torch.randn_like(x) * self.config.noise_std

        x = self._erase(x)
        return x.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)


def build_train_augment(config: dict[str, Any] | None) -> PairedAugment | None:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return None
    return PairedAugment(
        PairedAugmentConfig(
            enabled=True,
            horizontal_flip_p=float(cfg.get("horizontal_flip_p", 0.0)),
            crop_scale=tuple(float(x) for x in cfg.get("crop_scale", [1.0, 1.0])),
            brightness=float(cfg.get("brightness", 0.0)),
            contrast=float(cfg.get("contrast", 0.0)),
            saturation=float(cfg.get("saturation", 0.0)),
            gamma=tuple(float(x) for x in cfg.get("gamma", [1.0, 1.0])),
            color_shared=bool(cfg.get("color_shared", True)),
            noise_std=float(cfg.get("noise_std", 0.0)),
            erase_p=float(cfg.get("erase_p", 0.0)),
            erase_scale=tuple(float(x) for x in cfg.get("erase_scale", [0.02, 0.10])),
        )
    )
