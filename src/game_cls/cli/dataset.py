"""cls-trainer command line interface.

Workflows:

    cls-trainer train --config configs/npu_1p.yaml [key=value ...]
    cls-trainer train --config ... --dry-run
    cls-trainer train --resume <run_dir>
    cls-trainer config show --config ... [--with-source]
    cls-trainer config validate --config ...
    cls-trainer config reference
    cls-trainer run list [--root runs]
    cls-trainer run show latest|<run_dir>
    cls-trainer doctor --config ...

Every ``train`` start defaults to ``--run-mode unique``: the configured
``experiment.output_dir`` is treated as a runs root and a fresh timestamped
run directory is allocated, so re-running a command can never overwrite a
previous run. ``--run-mode fixed`` restores the legacy in-place behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# dataset prepare / audit / pack
# ---------------------------------------------------------------------------


def _run_split_prepare(
    config: dict[str, Any],
    train_all_root: str | Path,
    test_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict:
    """Derive train/val from train_all_root and write the split bundle.

    Thin wrapper around ``indexing.write_split_bundle``; the first argument
    is the physical ``source_root`` (train_all) whose frames are logically
    re-partitioned into train/val.
    """
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import write_split_bundle

    data_config = config["data"]
    split = dict(data_config.get("split") or {})
    if output_dir is None:
        # Match tools/build_index.py's default index output dir; the split
        # manifest is then written at output_dir/manifest (e.g.
        # indexes/split_manifest.parquet).
        output_dir = Path("indexes")
    return write_split_bundle(
        train_all_root,
        test_root,
        output_dir,
        ImageSpec.from_config(data_config),
        ScanPolicy.from_config(data_config),
        DuplicatePolicy.from_config(data_config),
        split_config=split,
        compute_content_hash=True,
    )


def _maybe_prepare_split(config: dict[str, Any]) -> None:
    """Build the validation split on demand when ``prepare_if_missing``."""
    data_config = config["data"]
    if not data_config.get("prepare_if_missing"):
        return
    split = data_config.get("split") or {}
    if split.get("mode") != "from_train":
        return
    val_index = data_config.get("val_index")
    if val_index and Path(val_index).is_file():
        return  # the validation split already exists
    source_root = data_config.get("source_root")
    test_root = data_config.get("test_root")
    if not source_root or not test_root:
        raise SystemExit(
            "data.prepare_if_missing requires data.source_root and "
            "data.test_root to derive the validation split; set both, or "
            "run 'cls-trainer dataset prepare' separately first."
        )
    _run_split_prepare(config, source_root, test_root)


def cmd_dataset_prepare(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    data_config = config["data"]
    split = data_config.get("split") or {}
    if split.get("mode") != "from_train":
        print(
            "dataset prepare requires data.split.mode == 'from_train'; "
            f"got {split.get('mode')!r}. Configure data.split (mode, "
            "val_ratio, seed) to derive a validation split.",
            file=sys.stderr,
        )
        return 2
    if args.val_ratio is not None:
        split["val_ratio"] = args.val_ratio
    audit = _run_split_prepare(
        config,
        args.train_root,
        args.test_root,
        output_dir=args.output_dir,
    )
    summary = audit.get("split") or audit
    print("=== dataset prepare finished ===")
    print(f"split mode        : {split.get('mode')}")
    print(f"val ratio target  : {split.get('val_ratio')}")
    print(f"val ratio achieved (delta=2): {summary.get('val_ratio_achieved_delta2')}")
    print(f"source videos     : {summary.get('source_video_count')}")
    print(f"manifest          : {split.get('manifest')}")
    return 0


def cmd_dataset_audit(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import audit_warning_messages, validate_audit

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    data_config = config["data"]
    source = Path(args.index_dir) / "audit.json"
    if not source.is_file():
        print(f"audit report not found: {source}", file=sys.stderr)
        return 2
    audit = json.loads(source.read_text(encoding="utf-8"))
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
        )
        print("Strict dataset audit passed.")
    return 0


def cmd_dataset_pack(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.packed_backend import pack_frame_index

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
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
    return 0


def cmd_dataset_annotate(args: argparse.Namespace) -> int:
    """Import per-video metadata into the sidecar (step5 P2).

    Reads a CSV/parquet keyed by ``source_video_uid``, validates every uid
    against the configured train/val/test video indexes, and writes the
    canonical sidecar parquet atomically.
    """
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.sidecar import (
        validate_sidecar_against_index,
        write_metadata_sidecar,
    )
    from game_cls.engine.training.loaders import _build_real_data_components

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2

    source = Path(args.metadata)
    if source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(source).to_pylist()
    else:
        import csv

        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [dict(row) for row in reader]
    if not rows:
        print(f"No metadata rows in {source}", file=sys.stderr)
        return 2
    if "source_video_uid" not in rows[0]:
        print(
            "Metadata input must contain a source_video_uid column.",
            file=sys.stderr,
        )
        return 2

    components = _build_real_data_components(config, rank=0, world_size=1)
    entries = (
        components["train_videos"]
        + components["val_videos"]
        + (components["test_videos"] or [])
    )
    sidecar = {
        str(row["source_video_uid"]): {
            "negative_subtype": row.get("negative_subtype"),
            "scene_type": row.get("scene_type"),
            "capture_domain": row.get("capture_domain"),
            "difficulty": row.get("difficulty"),
            "sample_weight": float(row.get("sample_weight", 1.0)),
        }
        for row in rows
    }
    validate_sidecar_against_index(sidecar, entries)

    out_path = Path(args.out) if args.out else Path(config["data"]["metadata_sidecar"])
    if not out_path:
        print(
            "No output sidecar path: pass --out or set data.metadata_sidecar.",
            file=sys.stderr,
        )
        return 2
    fingerprint = write_metadata_sidecar(rows, out_path)
    print(
        json.dumps(
            {
                "sidecar": str(out_path),
                "videos": len(sidecar),
                "metadata_fingerprint": fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
