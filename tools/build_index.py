from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.config import load_config
from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
from game_cls.data.indexing import write_index_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frame and video Parquet indexes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument(
        "--val-root",
        default=None,
        help=(
            "Validation split root. When omitted, training treats the test "
            "split as validation and no independent test set exists."
        ),
    )
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-dir", default="indexes")
    parser.add_argument(
        "--skip-content-hash",
        action="store_true",
        help="Skip SHA-256 leakage detection to reduce indexing I/O",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    data_config = config["data"]
    image_spec = ImageSpec.from_config(data_config)
    audit = write_index_bundle(
        train_root=args.train_root,
        test_root=args.test_root,
        output_dir=args.output_dir,
        image_spec=image_spec,
        scan_policy=ScanPolicy.from_config(data_config),
        duplicate_policy=DuplicatePolicy.from_config(data_config),
        val_root=args.val_root,
        compute_content_hash=not args.skip_content_hash,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
