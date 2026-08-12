from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_split(
    root: Path,
    split: str,
    *,
    videos_per_label: int,
    frames: int,
    size: tuple[int, int],
) -> None:
    height, width = size
    for label in (0, 1):
        directory = root / split / "game_a" / str(label)
        directory.mkdir(parents=True, exist_ok=True)
        for video in range(videos_per_label):
            for frame in range(frames):
                rng = np.random.default_rng(
                    10_000 * (split == "test") + label * 1000 + video * 100 + frame
                )
                base = 30 if label == 0 else 200
                array = np.clip(
                    base + rng.normal(0, 5, (height, width, 3)), 0, 255
                ).astype(np.uint8)
                Image.fromarray(array).save(directory / f"{video:02d}{frame:05d}.png")


def make_pair_dataset(
    root: Path,
    *,
    frames: int = 5,
    size: tuple[int, int] = (24, 32),
    train_videos_per_label: int = 4,
    test_videos_per_label: int = 2,
) -> None:
    _write_split(
        root,
        "train",
        videos_per_label=train_videos_per_label,
        frames=frames,
        size=size,
    )
    _write_split(
        root,
        "test",
        videos_per_label=test_videos_per_label,
        frames=frames,
        size=size,
    )
