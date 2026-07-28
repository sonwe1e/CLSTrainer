from __future__ import annotations

from typing import Any


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
            value = erasing.get("value", "random")
            transforms.append(
                v2.RandomErasing(
                    p=erasing.get("probability", 0.1),
                    scale=erasing.get("scale", [0.005, 0.03]),
                    ratio=erasing.get("ratio", [0.3, 3.3]),
                    value=value,
                )
            )
        self.transform = v2.Compose(transforms)

    def __call__(self, pair):
        return self.transform(pair)
