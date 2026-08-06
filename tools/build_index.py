from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.config import load_config
from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
from game_cls.data.indexing import write_index_bundle, write_split_bundle


def _split_config(data_config: dict) -> dict:
    """Read the ``data.split`` block as a standalone dict of literals."""
    split = data_config.get("split")
    if split is None:
        raise SystemExit(
            "--split-mode from_train requires a data.split block "
            "(data.split.mode: from_train, val_ratio, seed)."
        )
    return {
        "mode": split["mode"],
        "val_ratio": split["val_ratio"],
        "seed": split["seed"],
        "group_key": split.get("group_key", "source_video_uid"),
        "stratify_by": split.get("stratify_by", ["game", "label"]),
        "balance_by": split.get("balance_by", "legal_pair_count"),
        "target_delta": split.get("target_delta", 2),
        "manifest": split.get("manifest", "split_manifest.parquet"),
        "on_new_groups": split.get("on_new_groups", "error"),
        "small_stratum_policy": split.get("small_stratum_policy", "error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frame and video Parquet indexes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument(
        "--val-root",
        default=None,
        help=(
            "Validation split root. When omitted, training treats the test "
            "split as validation and no independent test set exists. "
            "Forbidden with --split-mode from_train."
        ),
    )
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-dir", default="indexes")
    parser.add_argument(
        "--split-mode",
        choices=("none", "from_train"),
        default="none",
        help=(
            "none (default): the existing three-root path via "
            "write_index_bundle. from_train: --train-root is the train_all "
            "root and the validation split is derived from it via "
            "write_split_bundle."
        ),
    )
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
    if args.split_mode == "from_train":
        if args.val_root is not None:
            parser.error(
                "--split-mode from_train derives the validation split from "
                "--train-root; --val-root is forbidden in this mode."
            )
        audit = write_split_bundle(
            train_all_root=args.train_root,
            test_root=args.test_root,
            output_dir=args.output_dir,
            image_spec=image_spec,
            scan_policy=ScanPolicy.from_config(data_config),
            duplicate_policy=DuplicatePolicy.from_config(data_config),
            split_config=_split_config(data_config),
            compute_content_hash=not args.skip_content_hash,
        )
    else:
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
