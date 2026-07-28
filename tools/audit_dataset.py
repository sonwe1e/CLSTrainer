from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.data.indexing import validate_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an index audit report")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--output-dir", default="reports/data_audit")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    source = Path(args.index_dir) / "audit.json"
    audit = json.loads(source.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "audit.json"
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for split, report in audit["splits"].items():
        print(
            f"{split}: frames={report['frame_count']} videos={report['video_count']} "
            f"unexpected_dimensions={report['unexpected_dimension_count']} "
            f"issues={len(report['parse_or_file_issues'])}"
        )
    if args.strict:
        validate_audit(audit)
        print("Strict dataset audit passed.")


if __name__ == "__main__":
    main()
