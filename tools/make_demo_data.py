from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _write_split(root: Path, split: str, *, videos: int, frames: int) -> None:
    height, width = 32, 48
    for label in (0, 1):
        for video in range(videos):
            directory = root / split / "demo_game" / str(label)
            directory.mkdir(parents=True, exist_ok=True)
            for frame in range(frames):
                rng = np.random.default_rng(
                    10_000 * (split == "test") + 1_000 * label + 100 * video + frame
                )
                base = 35 if label == 0 else 210
                image = np.clip(
                    base + rng.normal(0.0, 8.0, size=(height, width, 3)),
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(image).save(directory / f"{video:02d}{frame:05d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny CLSTrainer-Lite image dataset")
    parser.add_argument("--out", default="demo_data")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.out)
    _write_split(root, "train", videos=4, frames=args.frames)
    _write_split(root, "test", videos=2, frames=args.frames)
    print(f"demo dataset written to {root.resolve()}")


if __name__ == "__main__":
    main()
