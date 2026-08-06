from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from .records import FrameRecord, PairSample, build_pairs


def enumerate_pairs(
    frames: Sequence[FrameRecord], deltas: Sequence[int]
) -> list[PairSample]:
    pairs: list[PairSample] = []
    for delta in deltas:
        pairs.extend(build_pairs(frames, delta))
    return pairs


class PairDataset:
    """Map-style dataset; torch is imported only when a sample is decoded."""

    def __init__(
        self,
        pairs: Sequence[PairSample],
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
            from PIL import Image
            from torchvision.transforms.v2 import functional as F
        except ImportError as exc:
            raise RuntimeError(
                "Decoding training samples requires torch, torchvision and Pillow"
            ) from exc
        pair = self.pairs[index]
        with Image.open(pair.image0_path) as image0:
            tensor0 = F.to_image(image0.convert("RGB"))
        with Image.open(pair.image1_path) as image1:
            tensor1 = F.to_image(image1.convert("RGB"))
        images = torch.stack([tensor0, tensor1], dim=0)
        if self.transform is not None:
            images = self.transform(images)
        return {
            "images": images,
            "label": pair.label,
            "meta": pair,
        }


def group_pair_indices(
    pairs: Sequence[PairSample],
) -> dict[str, dict[int, dict[str, dict[int, list[int]]]]]:
    grouped: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for index, pair in enumerate(pairs):
        grouped[pair.game][pair.label][pair.video_id][pair.delta].append(index)
    return {
        game: {
            label: {
                video: dict(by_delta) for video, by_delta in by_video.items()
            }
            for label, by_video in by_label.items()
        }
        for game, by_label in grouped.items()
    }

