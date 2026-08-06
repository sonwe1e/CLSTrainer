from __future__ import annotations

import math
from typing import Any


def _identity(pair):
    return pair


class ConsistentUint8RandomErasing:
    """Random erasing with correct uint8 random-noise semantics."""

    def __init__(
        self,
        probability: float,
        scale,
        ratio,
        value: Any,
    ) -> None:
        self.probability = float(probability)
        self.scale = tuple(float(item) for item in scale)
        self.ratio = tuple(float(item) for item in ratio)
        self.value = value

    def __call__(self, pair):
        import torch

        if torch.rand(()) >= self.probability:
            return pair
        height, width = pair.shape[-2:]
        area = height * width
        erase_height = erase_width = None
        for _ in range(10):
            target_area = area * torch.empty(()).uniform_(*self.scale).item()
            aspect = math.exp(
                torch.empty(())
                .uniform_(math.log(self.ratio[0]), math.log(self.ratio[1]))
                .item()
            )
            candidate_height = int(round(math.sqrt(target_area * aspect)))
            candidate_width = int(round(math.sqrt(target_area / aspect)))
            if 0 < candidate_height <= height and 0 < candidate_width <= width:
                erase_height, erase_width = candidate_height, candidate_width
                break
        if erase_height is None or erase_width is None:
            return pair
        top = int(torch.randint(0, height - erase_height + 1, ()).item())
        left = int(torch.randint(0, width - erase_width + 1, ()).item())
        channels = pair.shape[-3]
        if self.value == "random":
            if pair.dtype == torch.uint8:
                fill = torch.randint(
                    0,
                    256,
                    (channels, erase_height, erase_width),
                    dtype=torch.uint8,
                    device=pair.device,
                )
            else:
                fill = torch.randn(
                    (channels, erase_height, erase_width),
                    dtype=pair.dtype,
                    device=pair.device,
                )
        elif isinstance(self.value, (list, tuple)):
            fill = torch.tensor(
                self.value, dtype=pair.dtype, device=pair.device
            ).reshape(channels, 1, 1)
        else:
            fill = self.value
        result = pair.clone()
        result[..., top : top + erase_height, left : left + erase_width] = fill
        return result


class _PairRandomResizedCrop:
    """Random-resized-crop sharing one crop across the stacked pair.

    The output ``size`` falls back to the input pair's own HxW when the
    ``size`` config key is omitted.
    """

    def __init__(self, v2, scale, ratio, interpolation) -> None:
        self.v2 = v2
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.interpolation = interpolation

    def __call__(self, pair):
        crop = self.v2.RandomResizedCrop(
            size=tuple(pair.shape[-2:]),
            scale=self.scale,
            ratio=self.ratio,
            interpolation=self.interpolation,
        )
        return crop(pair)


class _GammaPair:
    """One shared gamma draw applied to the whole stacked pair."""

    def __init__(self, probability: float, gamma_range) -> None:
        self.probability = float(probability)
        self.gamma_range = tuple(float(item) for item in gamma_range)

    def __call__(self, pair):
        import torch
        from torchvision.transforms.v2 import functional as F

        if torch.rand(()) >= self.probability:
            return pair
        gamma = torch.empty(()).uniform_(*self.gamma_range).item()
        return F.adjust_gamma(pair, gamma)


class _ExposurePair:
    """One shared multiplicative exposure factor applied to both frames."""

    def __init__(self, probability: float, factor_range) -> None:
        self.probability = float(probability)
        self.factor_range = tuple(float(item) for item in factor_range)

    def __call__(self, pair):
        import torch

        if torch.rand(()) >= self.probability:
            return pair
        factor = torch.empty(()).uniform_(*self.factor_range).item()
        if pair.dtype == torch.uint8:
            scaled = pair.to(torch.float32) * factor
            return torch.clamp(scaled, 0, 255).to(torch.uint8)
        return pair * factor


class _NoisePair:
    """One shared noise draw (identical values on both frames) added to the pair."""

    def __init__(self, probability: float, noise_std: float) -> None:
        self.probability = float(probability)
        self.noise_std = float(noise_std)

    def __call__(self, pair):
        import torch

        if torch.rand(()) >= self.probability:
            return pair
        noise = (
            torch.randn(pair.shape[1:], dtype=torch.float32, device=pair.device)
            * self.noise_std
        )
        if pair.dtype == torch.uint8:
            noisy = pair.to(torch.float32) + noise
            return torch.clamp(noisy, 0, 255).to(torch.uint8)
        return (pair.to(torch.float32) + noise).to(pair.dtype)


class _JpegCompressionPair:
    """One shared JPEG quality draw re-encoded onto both frames."""

    def __init__(self, probability: float, quality_range) -> None:
        self.probability = float(probability)
        self.quality_range = tuple(int(item) for item in quality_range)

    def __call__(self, pair):
        import io

        import torch
        from PIL import Image
        from torchvision.transforms.v2 import functional as F

        if torch.rand(()) >= self.probability:
            return pair
        quality = int(torch.empty(()).uniform_(*self.quality_range).item())
        if pair.dtype != torch.uint8:
            pair = torch.clamp(pair, 0, 255).to(torch.uint8)
        frames = []
        for frame in pair:
            buffer = io.BytesIO()
            F.to_pil_image(frame).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as image:
                frames.append(F.to_image(image))
        return torch.stack(frames, dim=0)


class ConsistentPairAugment:
    """Applies each randomly sampled transform once to a stacked [2,C,H,W] pair."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from torchvision.transforms import InterpolationMode, v2
        except ImportError as exc:
            raise RuntimeError("Augmentation requires torchvision") from exc

        transforms = []
        affine = config.get("random_affine", {})
        if affine.get("enabled", False):
            transforms.append(
                v2.RandomApply(
                    [
                        v2.RandomAffine(
                            degrees=affine.get("degrees", 2.0),
                            translate=affine.get("translate", [0.02, 0.02]),
                            scale=affine.get("scale", [0.98, 1.02]),
                            shear=affine.get("shear", [-1.0, 1.0]),
                            interpolation=InterpolationMode[
                                str(affine.get("interpolation", "bilinear")).upper()
                            ],
                            fill=affine.get("fill", 0),
                        )
                    ],
                    p=affine.get("probability", 0.5),
                )
            )
        color = config.get("color_jitter", {})
        if color.get("enabled", False):
            transforms.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=color.get("brightness", 0.15),
                            contrast=color.get("contrast", 0.15),
                            saturation=color.get("saturation", 0.10),
                            hue=color.get("hue", 0.02),
                        )
                    ],
                    p=color.get("probability", 0.8),
                )
            )
        erasing = config.get("random_erasing", {})
        if erasing.get("enabled", False):
            transforms.append(
                ConsistentUint8RandomErasing(
                    probability=erasing.get("probability", 0.1),
                    scale=erasing.get("scale", [0.005, 0.03]),
                    ratio=erasing.get("ratio", [0.3, 3.3]),
                    value=erasing.get("value", 0),
                )
            )
        perspective = config.get("random_perspective", {})
        if perspective.get("enabled", False):
            transforms.append(
                v2.RandomPerspective(
                    distortion_scale=perspective.get("distortion_scale", 0.2),
                    p=perspective.get("probability", 0.0),
                )
            )
        crop_cfg = config.get("random_resized_crop", {})
        if crop_cfg.get("enabled", False):
            crop_size = crop_cfg.get("size")
            if crop_size is None:
                crop_transform = _PairRandomResizedCrop(
                    v2,
                    scale=crop_cfg.get("scale", [0.8, 1.0]),
                    ratio=crop_cfg.get("ratio", [0.75, 1.3333]),
                    interpolation=InterpolationMode.BILINEAR,
                )
            else:
                crop_transform = v2.RandomResizedCrop(
                    size=tuple(crop_size),
                    scale=tuple(crop_cfg.get("scale", [0.8, 1.0])),
                    ratio=tuple(crop_cfg.get("ratio", [0.75, 1.3333])),
                    interpolation=InterpolationMode.BILINEAR,
                )
            transforms.append(
                v2.RandomApply(
                    [crop_transform],
                    p=crop_cfg.get("probability", 0.0),
                )
            )
        gamma = config.get("gamma", {})
        if gamma.get("enabled", False):
            transforms.append(
                _GammaPair(
                    probability=gamma.get("probability", 0.0),
                    gamma_range=gamma.get("gamma_range", [0.8, 1.2]),
                )
            )
        exposure = config.get("exposure", {})
        if exposure.get("enabled", False):
            transforms.append(
                _ExposurePair(
                    probability=exposure.get("probability", 0.0),
                    factor_range=exposure.get("factor_range", [0.85, 1.15]),
                )
            )
        blur = config.get("blur", {})
        if blur.get("enabled", False):
            transforms.append(
                v2.RandomApply(
                    [
                        v2.GaussianBlur(
                            kernel_size=blur.get("kernel_size", 3),
                            sigma=tuple(blur.get("sigma_range", [0.1, 1.0])),
                        )
                    ],
                    p=blur.get("probability", 0.0),
                )
            )
        noise = config.get("noise", {})
        if noise.get("enabled", False):
            transforms.append(
                _NoisePair(
                    probability=noise.get("probability", 0.0),
                    noise_std=noise.get("noise_std", 0.02),
                )
            )
        jpeg = config.get("jpeg_compression", {})
        if jpeg.get("enabled", False):
            transforms.append(
                _JpegCompressionPair(
                    probability=jpeg.get("probability", 0.0),
                    quality_range=jpeg.get("quality_range", [60, 95]),
                )
            )
        # v2.Compose rejects an empty list; with every transform disabled the
        # augmenter must be a no-op (identity) rather than raise.
        self.transform = v2.Compose(transforms) if transforms else _identity

    def __call__(self, pair):
        return self.transform(pair)
