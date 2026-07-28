from __future__ import annotations

import math
from typing import Any


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
                torch.empty(()).uniform_(
                    math.log(self.ratio[0]), math.log(self.ratio[1])
                ).item()
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


class ConsistentPairAugment:
    """Applies each randomly sampled transform once to a stacked [2,C,H,W] pair."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from torchvision.transforms import v2
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
        self.transform = v2.Compose(transforms)

    def __call__(self, pair):
        return self.transform(pair)
