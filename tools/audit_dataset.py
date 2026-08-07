from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.config import load_config
from game_cls.config_schema import resolve_source_identity_namespaces
from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
from game_cls.data.indexing import audit_warning_messages, validate_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an index audit report")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--output-dir", default="reports/data_audit")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    data_config = config["data"]
    namespaces_by_split = resolve_source_identity_namespaces(
        data_config.get("source_video_identity")
    )
    source = Path(args.index_dir) / "audit.json"
    audit = json.loads(source.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "audit.json"
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for split, report in audit["splits"].items():
        findings = report.get("findings", {})
        print(
            f"{split}: frames={report['frame_count']} videos={report['video_count']} "
            f"errors={len(findings.get('errors', []))} "
            f"warnings={len(findings.get('warnings', []))} "
            f"ignored={sum(findings.get('ignored', {}).get('counts', {}).values())}"
        )
    for warning in audit_warning_messages(audit):
        print(f"[WARNING] {warning}")
    if args.strict:
        validate_audit(
            audit,
            image_spec=ImageSpec.from_config(data_config),
            scan_policy=ScanPolicy.from_config(data_config),
            duplicate_policy=DuplicatePolicy.from_config(data_config),
            require_test_delta=int(config["pair"]["test_delta"]),
            require_content_hash=bool(
                data_config.get("require_content_hash_audit", False)
            ),
            require_unique_video_keys=bool(
                data_config.get("require_unique_video_keys_across_splits", False)
            ),
            minimum_pairs_per_game_label_delta={
                int(key): int(value)
                for key, value in data_config.get(
                    "minimum_pairs_per_game_label_delta", {}
                ).items()
            },
            identity_mode=(data_config.get("source_video_identity") or {}).get(
                "mode", "game_video"
            ),
            namespaces_by_split=namespaces_by_split,
        )
        print("Strict dataset audit passed.")


if __name__ == "__main__":
    main()
