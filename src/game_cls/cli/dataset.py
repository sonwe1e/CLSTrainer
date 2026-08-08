"""cls-trainer command line interface.

Workflows:

    cls-trainer train --config configs/recipes/game_cls_production.yaml [key=value ...]
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


def _resolve_source_video_index(
    frame_index: str | Path | None,
    explicit: str | Path | None = None,
) -> Path | None:
    """The source video-entry parquet to inherit canonical uid from (audit P0-5).

    Precedence:
    1. ``explicit`` -- the operator said so with ``--source-video-index``.
    2. The sibling ``*_video_entries.parquet`` of ``frame_index`` (e.g.
       ``indexes/train_frames.parquet`` -> ``indexes/train_video_entries.parquet``).
    3. ``None`` -- uid falls back to ``game::label::video_id``.
    """
    if explicit:
        return Path(explicit)
    if not frame_index:
        return None
    stem = Path(frame_index).stem  # e.g. "train_frames"
    if stem.endswith("_frames"):
        candidate = Path(frame_index).with_name(
            stem[: -len("_frames")] + "_video_entries.parquet"
        )
        if candidate.is_file():
            return candidate
    return None

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
    from game_cls.config_schema import resolve_source_identity_namespaces
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import write_split_bundle

    data_config = config["data"]
    split = dict(data_config.get("split") or {})
    source_video_identity = data_config.get("source_video_identity") or {}
    identity_mode = source_video_identity.get("mode", "game_video")
    if identity_mode == "game_label_video":
        print(
            "[WARNING] source_video_identity.mode=game_label_video\n"
            "The framework assumes identical video_id values under different labels "
            "are\n"
            "physically unrelated source videos. If this assumption is false, "
            "train/validation\n"
            "leakage may occur."
        )
    namespaces_by_split = resolve_source_identity_namespaces(source_video_identity)
    if namespaces_by_split:
        print(
            "[INFO] source identity namespaces: "
            + ", ".join(
                f"{split}={namespace}"
                for split, namespace in sorted(namespaces_by_split.items())
            )
        )
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
        identity_mode=identity_mode,
        namespaces_by_split=namespaces_by_split,
    )


def _split_index_dir(data_config: dict[str, Any]) -> Path:
    """The index directory a split bundle is published into."""
    return Path(data_config.get("train_index") or "indexes/train_frames.parquet").parent


def _split_manifest_path(data_config: dict[str, Any]) -> Path | None:
    """``data.split.manifest``, resolved relative to the index dir when needed."""
    manifest = (data_config.get("split") or {}).get("manifest")
    if not manifest:
        return None
    path = Path(manifest)
    if not path.is_absolute():
        path = _split_index_dir(data_config) / path
    return path


def _split_bundle_artifacts(data_config: dict[str, Any]) -> dict[str, str]:
    """Display label -> path for every artifact a complete split bundle produces.

    Checking only ``val_index`` was not enough: a prepare that died partway
    (or a partially copied index directory) leaves val_frames.parquet on disk
    while the video-entry parquets are missing, and training then fails much
    later with a confusing read error.

    Audit P0-9 widened this to the split manifest and split summary, which the
    bundle also publishes and which were previously unchecked.

    Existence is all this reports. It deliberately cannot detect a *mixed*
    generation (train from a new prepare, test left from an older one), because
    every file exists in exactly that case -- that is what the bundle manifest
    and ``indexing.verify_split_bundle`` are for. This answers the narrower
    question "was anything built at all".
    """
    keys = (
        "train_index",
        "val_index",
        "test_index",
        "train_video_index",
        "val_video_index",
        "test_video_index",
        "audit_path",
    )
    artifacts: dict[str, str] = {
        f"data.{key}": str(data_config[key]) for key in keys if data_config.get(key)
    }
    manifest_path = _split_manifest_path(data_config)
    if manifest_path is not None:
        artifacts["data.split.manifest"] = str(manifest_path)
    artifacts["split_summary"] = str(
        _split_index_dir(data_config) / "split_summary.json"
    )
    return artifacts


def _missing_split_artifacts(data_config: dict[str, Any]) -> list[str]:
    return [
        f"{label}={path}"
        for label, path in _split_bundle_artifacts(data_config).items()
        if not Path(path).is_file()
    ]


def _maybe_prepare_split(config: dict[str, Any]) -> None:
    """Build the validation split on demand when ``prepare_if_missing``.

    Runs before the distributed runtime exists, so every rank of a torchrun
    launch reaches this point concurrently. There is no lock and no atomic
    directory commit, so eight ranks scanning and writing the same index
    files would interleave partial writes. Rather than invent a locking
    protocol here, a multi-process launch refuses to prepare and tells the
    operator to run ``cls-trainer dataset prepare`` once, up front.
    """
    import os

    data_config = config["data"]
    if not data_config.get("prepare_if_missing"):
        return
    split = data_config.get("split") or {}
    if split.get("mode") != "from_train":
        return
    missing = _missing_split_artifacts(data_config)
    if not missing:
        # Audit P0-9: "nothing missing" is NOT "one consistent bundle" -- a
        # mixed generation (train from a new prepare, test left from an older
        # one) satisfies every existence check. Verify the commit record before
        # treating the bundle as usable, or prepare_if_missing silently skips
        # straight into training on a mixed split.
        from game_cls.data.indexing import SplitBundleError, verify_split_bundle

        try:
            verify_split_bundle(_split_index_dir(data_config))
        except SplitBundleError as exc:
            raise SystemExit(str(exc)) from exc
        return  # the whole split bundle already exists and is one generation

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size > 1:
        rank = os.environ.get("RANK", "?")
        raise SystemExit(
            "data.prepare_if_missing cannot run under a multi-process launch "
            f"(WORLD_SIZE={world_size}, RANK={rank}): every rank would scan "
            "the dataset and write the same index files concurrently, with no "
            "lock and no atomic commit. Prepare once first:\n"
            "    cls-trainer dataset prepare --config <config> "
            "--train-root <train_all> --test-root <test>\n"
            "then relaunch training. Missing artifacts: " + ", ".join(missing)
        )

    source_root = data_config.get("source_root")
    test_root = data_config.get("test_root")
    if not source_root or not test_root:
        raise SystemExit(
            "data.prepare_if_missing requires data.source_root and "
            "data.test_root to derive the validation split; set both, or "
            "run 'cls-trainer dataset prepare' separately first. Missing "
            "artifacts: " + ", ".join(missing)
        )
    # Audit PR-C: write into the CONFIGURED index directory, not the
    # CWD-relative ./indexes default -- otherwise a config pointing
    # data.train_index at /data/project/indexes_v2 would "prepare successfully"
    # while the configured paths stayed missing and training still refused.
    configured_index_dir = Path(
        data_config.get("train_index", "indexes/train_frames.parquet")
    ).parent
    _run_split_prepare(
        config, source_root, test_root, output_dir=str(configured_index_dir)
    )


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
    delta = summary.get("target_delta", split.get("target_delta", 2))
    achieved = summary.get("val_ratio_achieved")
    if "source_identity_precheck" in audit:
        from game_cls.data.splitter import format_source_identity_precheck

        print(format_source_identity_precheck(audit["source_identity_precheck"]))
    print("=== dataset prepare finished ===")
    print(f"split mode        : {split.get('mode')}")
    print(f"val ratio target  : {split.get('val_ratio')}")
    print(f"val ratio achieved (delta={delta}): {achieved}")
    print(f"source videos     : {summary.get('source_video_count')}")
    print(f"manifest          : {split.get('manifest')}")
    if summary.get("manifest_extended"):
        print(f"extended with     : {summary.get('added_source_videos')} source videos")
    if not summary.get("fingerprint_covers_content", True):
        print(
            "note              : content hashing was off, so the split "
            "fingerprint cannot detect a same-size frame edit"
        )
    return 0


def cmd_dataset_audit(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import (
        ConfigSchemaError,
        resolve_source_identity_namespaces,
    )
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import (
        audit_warning_messages,
        format_uid_overlap_content_classification,
        validate_audit,
    )

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    data_config = config["data"]
    source_video_identity = data_config.get("source_video_identity") or {}
    namespaces_by_split = resolve_source_identity_namespaces(source_video_identity)
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
    # Print the collision classification before the strict gate so the
    # diagnostic stays visible when the gate fails.
    classification = audit.get("leakage", {}).get(
        "source_uid_overlap_content_classification"
    )
    if classification:
        print(format_uid_overlap_content_classification(classification))
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
            identity_mode=source_video_identity.get("mode", "game_video"),
            namespaces_by_split=namespaces_by_split,
        )
        print("Strict dataset audit passed.")
    return 0


def cmd_dataset_seal(args: argparse.Namespace) -> int:
    """Adopt an existing index directory as one bundle generation (audit P0-9).

    Bundles built before the transactional writer carry no
    ``bundle_manifest.json``, so nothing can rule out that they are a mixed
    generation -- and the read path refuses them rather than guessing. Rebuilding
    with ``dataset prepare`` is the safe route; this command is the alternative
    for an operator who knows the directory is one consistent generation.

    That knowledge is genuinely the operator's: the framework cannot infer it
    from files that all exist. So sealing is explicit, and it refuses when an
    artifact is missing rather than committing a known-incomplete bundle.
    """
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.indexing import (
        BUNDLE_ARTIFACT_NAMES,
        SplitBundleError,
        verify_split_bundle,
        write_bundle_manifest,
    )

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    data_config = config["data"]
    index_dir = (
        Path(args.index_dir) if args.index_dir else _split_index_dir(data_config)
    )
    if not index_dir.is_dir():
        print(f"Index directory not found: {index_dir}", file=sys.stderr)
        return 2

    absent = [
        name for name in BUNDLE_ARTIFACT_NAMES if not (index_dir / name).is_file()
    ]
    if absent:
        print(
            f"Refusing to seal {index_dir}: it is missing {', '.join(absent)}. "
            "Sealing would commit a bundle that is already known to be "
            "incomplete. Rebuild it instead:\n"
            "    cls-trainer dataset prepare --config <config> "
            "--train-root <train_all> --test-root <test>",
            file=sys.stderr,
        )
        return 2

    # A pre-existing bundle_id inside audit.json/split_summary.json would
    # disagree with the new manifest and read as a mixed generation, so restamp
    # both to the generation being committed now.
    from uuid import uuid4

    bundle_id = uuid4().hex
    for name in ("audit.json", "split_summary.json"):
        path = index_dir / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(
                f"Cannot seal {index_dir}: {name} is unreadable: {exc}", file=sys.stderr
            )
            return 2
        if not isinstance(payload, dict):
            print(
                f"Cannot seal {index_dir}: {name} is not a JSON object.",
                file=sys.stderr,
            )
            return 2
        payload["bundle_id"] = bundle_id
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # The split manifest is not hash-committed here either -- write_split_bundle
    # leaves it out because resolve_split mutates it before the parquets are
    # staged, and seal must produce a manifest the same shape or a sealed bundle
    # and a prepared one would verify under different rules.
    manifest = write_bundle_manifest(index_dir, bundle_id)
    try:
        verify_split_bundle(index_dir)
    except SplitBundleError as exc:
        # Sealing then failing verification means the directory changed under
        # us; surface it rather than reporting a success that does not hold.
        print(f"Sealed {manifest} but verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "index_dir": str(index_dir),
                "bundle_id": bundle_id,
                "bundle_manifest": str(manifest),
                "artifacts": len(BUNDLE_ARTIFACT_NAMES),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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
        # Audit P0-6: bind the shards to the exact index/audit/split-manifest
        # they were generated from, so a regenerated index with stale shards is
        # refused at DataLoader creation.
        audit_path=config["data"].get("audit_path"),
        split_manifest_path=(config["data"].get("split") or {}).get("manifest"),
        # Audit P0-5: inherit canonical_source_video_uid and sidecar fields so
        # the packed backend and the PNG backend agree on identity. Without this
        # the packed video index falls back to game::label::video_id while the
        # source index carries game::label::video_id#content_hash, and every
        # sidecar/mining/subtype join keyed on the uid silently drifts.
        #
        # Precedence: explicit --source-video-index, then auto-derive from the
        # sibling *_video_entries.parquet of --frame-index (covers the common
        # case where the user points at train_frames.parquet without having to
        # add a second flag), then None (uid falls back, packing still works).
        source_video_index=_resolve_source_video_index(
            args.frame_index,
            getattr(args, "source_video_index", None),
        ),
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


# ---------------------------------------------------------------------------
# dataset annotate
# ---------------------------------------------------------------------------

# Sidecar columns an import may set. Anything else in the input is ignored;
# write_metadata_sidecar normalizes to the parquet schema anyway.
_SIDECAR_FIELDS = (
    "negative_subtype",
    "scene_type",
    "capture_domain",
    "difficulty",
    "sample_weight",
)


def _annotate_field(row: dict[str, Any], field: str) -> Any:
    """Value of ``field`` when the input really supplies one, else ``None``.

    A missing key, ``None`` and a blank string (an empty CSV cell) all mean
    "not specified", so merging keeps whatever the sidecar already held
    instead of blanking a hand-made annotation.
    """
    value = row.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if field == "sample_weight":
        return float(value)
    return str(value)


def _mining_rows(mined: list[dict[str, Any]], subtype: str) -> list[dict[str, Any]]:
    """Collapse mining top-K *pairs* into one row per source video.

    A mining manifest holds up to ``top_k_per_video`` rows per video, but
    ``source_video_uid`` is the sidecar's primary key, so the rows must be
    aggregated before they can be written. The representative is the
    highest-``p_positive`` pair: that score is the video's hardest pair, which
    is the honest per-video difficulty signal and is exactly how
    ``scan_negative_pool`` already ranks candidates. Picking the max is also
    order-independent, so re-reading the same manifest cannot flip the result
    the way "first row wins" would.

    ``p_positive`` rides along on the representative row (the sidecar schema
    has no score column, so it is ignored downstream) purely so the selection
    stays auditable and unit-testable.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in mined:
        uid = str(row["source_video_uid"])
        score = float(row.get("p_positive", 0.0))
        current = best.get(uid)
        if current is None or score > current["p_positive"]:
            best[uid] = {
                "source_video_uid": uid,
                "negative_subtype": subtype,
                "p_positive": score,
            }
    return [best[uid] for uid in sorted(best)]


def _read_metadata_input(source: Path) -> list[dict[str, Any]]:
    """Read the per-video annotation table (parquet or CSV)."""
    if source.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        # memory_map=False keeps no Windows file handle on the input, which
        # matters when the input and the sidecar live in the same temp dir.
        return pq.read_table(source, memory_map=False).to_pylist()
    import csv

    with source.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _merge_sidecar_rows(
    existing: dict[str, dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    on_subtype_conflict: str,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, str]], dict[str, int]]:
    """Merge an import into the existing sidecar, per field.

    Annotate is additive: importing mined negatives must not destroy the
    subtypes, weights and scene tags someone entered by hand. Fields the
    import does not specify are carried over untouched, and a uid that the
    import never mentions keeps its row.
    """
    merged = {uid: dict(meta) for uid, meta in existing.items()}
    conflicts: list[tuple[str, str, str]] = []
    stats = {"new": 0, "updated": 0}
    for row in incoming:
        uid = str(row["source_video_uid"])
        before = merged.get(uid)
        if before is None:
            new_row = {field: _annotate_field(row, field) for field in _SIDECAR_FIELDS}
            # A null sample_weight would break both readers, so the column is
            # always a real float.
            if new_row["sample_weight"] is None:
                new_row["sample_weight"] = 1.0
            merged[uid] = new_row
            stats["new"] += 1
            continue
        after = dict(before)
        for field in _SIDECAR_FIELDS:
            value = _annotate_field(row, field)
            if value is None:
                continue  # absent in the import: keep the current value
            if field == "negative_subtype":
                current = before.get("negative_subtype")
                if current and current != value:
                    conflicts.append((uid, str(current), str(value)))
                    # refuse aborts before anything is written; keep leaves the
                    # human annotation in place. Only overwrite falls through.
                    if on_subtype_conflict in ("refuse", "keep"):
                        continue
            after[field] = value
        if after.get("sample_weight") is None:
            after["sample_weight"] = 1.0
        if after != before:
            stats["updated"] += 1
        merged[uid] = after
    return merged, conflicts, stats


def cmd_dataset_annotate(args: argparse.Namespace) -> int:
    """Import per-video metadata into the sidecar (step5 P2).

    Reads a CSV/parquet keyed by ``source_video_uid`` (or a mining manifest
    via ``--from-mining``), validates every uid against the configured
    train/val/test video indexes, merges the rows into whatever sidecar
    already exists and rewrites it atomically.
    """
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.sidecar import (
        read_metadata_sidecar,
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

    mining_pairs: int | None = None
    if args.from_mining:
        from game_cls.reports.benchmark import read_mining_manifest

        if not args.subtype:
            print("--from-mining requires --subtype.", file=sys.stderr)
            return 2
        mined = read_mining_manifest(args.from_mining)
        mining_pairs = len(mined)
        rows = _mining_rows(mined, args.subtype)
        if not rows:
            print(f"No mined negatives in {args.from_mining}", file=sys.stderr)
            return 2
    else:
        source = Path(args.metadata)
        rows = _read_metadata_input(source)
        if not rows:
            print(f"No metadata rows in {source}", file=sys.stderr)
            return 2
        if "source_video_uid" not in rows[0]:
            print(
                "Metadata input must contain a source_video_uid column.",
                file=sys.stderr,
            )
            return 2

    # Resolve the destination before touching the indexes: data.metadata_sidecar
    # defaults to null, and Path(None) used to raise a TypeError long before the
    # friendly message below could be printed.
    configured_out = config["data"].get("metadata_sidecar")
    out = args.out or configured_out
    if not out:
        print(
            "No output sidecar path: pass --out or set data.metadata_sidecar.",
            file=sys.stderr,
        )
        return 2
    out_path = Path(out)

    components = _build_real_data_components(config, rank=0, world_size=1)
    entries = (
        components["train_videos"]
        + components["val_videos"]
        + (components["test_videos"] or [])
    )
    incoming = {
        str(row["source_video_uid"]): {
            field: _annotate_field(row, field) for field in _SIDECAR_FIELDS
        }
        for row in rows
    }
    # Only the imported uids are checked: rows already in the sidecar were
    # validated when they were written, and failing an operator's import over
    # a stale pre-existing row they did not touch would be unhelpful.
    validate_sidecar_against_index(incoming, entries)

    merged, conflicts, stats = _merge_sidecar_rows(
        read_metadata_sidecar(out_path),
        rows,
        on_subtype_conflict=args.on_subtype_conflict,
    )
    if conflicts and args.on_subtype_conflict == "refuse":
        preview = ", ".join(
            f"{uid} ({current} -> {incoming_subtype})"
            for uid, current, incoming_subtype in conflicts[:10]
        )
        print(
            f"Refusing to overwrite {len(conflicts)} existing negative_subtype "
            f"annotation(s) in {out_path}: {preview}"
            f"{'...' if len(conflicts) > 10 else ''}\n"
            "Re-run with --on-subtype-conflict keep (retain the existing "
            "annotation) or overwrite (take the import).",
            file=sys.stderr,
        )
        return 2

    sidecar_rows = [
        {"source_video_uid": uid, **meta} for uid, meta in sorted(merged.items())
    ]
    fingerprint = write_metadata_sidecar(sidecar_rows, out_path)
    payload = {
        "sidecar": str(out_path),
        "videos": len(sidecar_rows),
        "imported": len(incoming),
        "new": stats["new"],
        "updated": stats["updated"],
        "subtype_conflicts": len(conflicts),
        "on_subtype_conflict": args.on_subtype_conflict,
        "metadata_fingerprint": fingerprint,
    }
    if mining_pairs is not None:
        # Makes the top-K -> one-row-per-video aggregation visible.
        payload["mining_pairs"] = mining_pairs
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
