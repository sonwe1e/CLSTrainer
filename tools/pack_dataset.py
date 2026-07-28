from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.data.packed_backend import pack_frame_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack decoded CHW uint8 images into memory-mapped shards"
    )
    parser.add_argument("--frame-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-per-shard", type=int, default=4096)
    parser.add_argument("--width", type=int, default=208)
    parser.add_argument("--height", type=int, default=448)
    args = parser.parse_args()
    index_path = pack_frame_index(
        args.frame_index,
        args.output_dir,
        images_per_shard=args.images_per_shard,
        expected_width=args.width,
        expected_height=args.height,
    )
    print(index_path)


if __name__ == "__main__":
    main()
