from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.config import load_config
from game_cls.data.image_spec import ImageSpec
from game_cls.data.packed_backend import pack_frame_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack decoded CHW uint8 images into memory-mapped shards"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-per-shard", type=int, default=4096)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    index_path = pack_frame_index(
        args.frame_index,
        args.output_dir,
        image_spec=ImageSpec.from_config(config["data"]),
        images_per_shard=args.images_per_shard,
    )
    print(
        json.dumps(
            {
                "packed_frame_index": str(index_path),
                "packed_video_index": str(
                    index_path.with_name("packed_video_entries.parquet")
                ),
                "manifest": str(index_path.with_name("packed_manifest.json")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
