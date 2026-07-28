from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.data.indexing import write_index_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frame and video Parquet indexes")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-dir", default="indexes")
    args = parser.parse_args()
    audit = write_index_bundle(args.train_root, args.test_root, args.output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

