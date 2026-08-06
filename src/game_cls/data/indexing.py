from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .image_spec import ImageSpec
from .index_policy import DuplicatePolicy, ScanFindings, ScanPolicy
from .records import (
    DEFAULT_FILENAME_PATTERN,
    FrameRecord,
    VideoRecord,
    parse_filename,
    read_png_metadata,
    summarize_videos,
)
from .splitter import resolve_split, write_split_summary

AUDIT_FORMAT_VERSION = 3

# Ordered split roles of the train/validation/test protocol.
SPLIT_ORDER = ("train", "val", "test")


def source_video_uid(game: str, video_id: str) -> str:
    """Stable, label-independent identity of a source video.

    Two-digit ``video_id`` values alone are not globally unique; the
    game prefix makes them safe for cross-split leakage checks. Adjacent
    frames and every delta pair of one source video must live in a single
    split, so this identity is the unit of leakage detection.
    """
    return f"{game}::{video_id}"


@dataclass(frozen=True)
class ScanResult:
    frames: list[FrameRecord]
    findings: ScanFindings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_frame_candidates(
    root: Path,
    policy: ScanPolicy,
    findings: ScanFindings,
) -> Iterator[tuple[str, int, Path]]:
    """Yield frames directly under <game>/<0|1>, pruning ignored content."""

    for game_dir in sorted(root.iterdir()):
        if game_dir.is_dir() and policy.ignore_directory(game_dir):
            findings.add_ignored("ignored_directory", game_dir)
            continue
        if not game_dir.is_dir():
            findings.add_ignored("non_directory_at_game_level", game_dir)
            continue

        for label_dir in sorted(game_dir.iterdir()):
            if label_dir.is_dir() and policy.ignore_directory(label_dir):
                findings.add_ignored("ignored_directory", label_dir)
                continue
            if not label_dir.is_dir():
                findings.add_ignored(
                    "non_directory_at_label_level", label_dir
                )
                continue
            if label_dir.name not in {"0", "1"}:
                candidate_frames = [
                    item
                    for item in label_dir.iterdir()
                    if item.is_file()
                    and not policy.ignore_file(item)
                    and policy.is_frame_file(item)
                ]
                if candidate_frames:
                    findings.add(
                        "error",
                        "invalid_label_directory",
                        label_dir,
                        "Directory containing frame files must be named 0 or 1",
                    )
                else:
                    findings.add_ignored(
                        "non_label_directory_without_frames", label_dir
                    )
                continue

            for item in sorted(label_dir.iterdir()):
                if item.is_dir():
                    if policy.ignore_directory(item):
                        findings.add_ignored("ignored_directory", item)
                    else:
                        findings.add(
                            policy.unexpected_nested_directory_severity,
                            "unexpected_nested_directory",
                            item,
                            "Frames must be directly below <game>/<0|1>",
                        )
                    continue
                if not item.is_file():
                    continue
                if policy.ignore_file(item):
                    findings.add_ignored("ignored_file_pattern", item)
                    continue
                if not policy.is_frame_file(item):
                    findings.add_ignored("non_frame_extension", item)
                    continue
                yield game_dir.name, int(label_dir.name), item


def scan_split(
    root: str | Path,
    split: str,
    image_spec: ImageSpec,
    *,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    scan_policy: ScanPolicy,
    compute_content_hash: bool = True,
) -> ScanResult:
    root = Path(root).resolve()
    image_spec.validate()
    findings = ScanFindings(
        ignored_example_limit=scan_policy.ignored_example_limit
    )
    frames: list[FrameRecord] = []
    if not root.is_dir():
        raise FileNotFoundError(f"{split} root does not exist: {root}")

    for game, label, path in iter_frame_candidates(
        root, scan_policy, findings
    ):
        try:
            video_id, frame_id = parse_filename(path.name, filename_pattern)
        except ValueError as exc:
            findings.add("error", "invalid_filename", path, str(exc))
            continue
        try:
            width, height, channels = read_png_metadata(path)
        except (OSError, ValueError) as exc:
            findings.add("error", "invalid_file", path, str(exc))
            continue
        if (width, height, channels) != (
            image_spec.width,
            image_spec.height,
            image_spec.channels,
        ):
            findings.add(
                "error",
                "unexpected_dimensions",
                path,
                (
                    f"expected {image_spec.width}x{image_spec.height}x"
                    f"{image_spec.channels}, got {width}x{height}x{channels}"
                ),
            )
            continue
        try:
            file_size = path.stat().st_size
            content_sha256 = (
                _sha256(path) if compute_content_hash else ""
            )
        except OSError as exc:
            findings.add("error", "unreadable_file", path, str(exc))
            continue
        label_text = str(label)
        frames.append(
            FrameRecord(
                sample_id=(
                    f"{split}:{game}:{label_text}:{video_id}:{frame_id:05d}"
                ),
                split=split,
                game=game,
                label=label,
                video_id=video_id,
                frame_id=frame_id,
                path=str(path),
                width=width,
                height=height,
                channels=channels,
                file_size=file_size,
                content_sha256=content_sha256,
            )
        )
    return ScanResult(frames=frames, findings=findings)


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet indexes requires pyarrow: "
            "python -m pip install pyarrow"
        ) from exc
    return pa, pq


def write_parquet(
    records: Iterable[FrameRecord | VideoRecord], path: str | Path
) -> None:
    pa, pq = _pyarrow()
    rows = [asdict(record) for record in records]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def read_frame_parquet(path: str | Path) -> list[FrameRecord]:
    _, pq = _pyarrow()
    return [
        FrameRecord(**row) for row in pq.read_table(path).to_pylist()
    ]


def analyze_content_duplicates(
    frames_by_split: dict[str, list[FrameRecord]],
    policy: DuplicatePolicy,
) -> dict:
    by_hash: dict[str, list[FrameRecord]] = defaultdict(list)
    by_basename: dict[str, list[FrameRecord]] = defaultdict(list)
    all_frames: list[FrameRecord] = []
    for frames in frames_by_split.values():
        for frame in frames:
            all_frames.append(frame)
            by_basename[Path(frame.path).name].append(frame)
            if frame.content_sha256:
                by_hash[frame.content_sha256].append(frame)

    result: dict[str, object] = {
        "errors": [],
        "warnings": [],
        "info": [],
        "content_hash_check_enabled": bool(all_frames)
        and all(bool(frame.content_sha256) for frame in all_frames),
    }
    for sha256, records in sorted(by_hash.items()):
        if len(records) < 2:
            continue
        labels = {record.label for record in records}
        splits = {record.split for record in records}
        payload = {
            "sha256": sha256,
            "count": len(records),
            "labels": sorted(labels),
            "splits": sorted(splits),
            "samples": [
                {
                    "split": record.split,
                    "game": record.game,
                    "label": record.label,
                    "video_id": record.video_id,
                    "frame_id": record.frame_id,
                    "path": record.path,
                }
                for record in records[:20]
            ],
        }
        if len(labels) > 1:
            severity = policy.cross_label_same_content
            kind = "identical_content_with_conflicting_labels"
        elif len(splits) > 1:
            severity = policy.same_label_cross_split
            kind = "same_label_content_overlap_across_splits"
        else:
            severity = policy.same_label_within_split
            kind = "duplicate_content_within_split"
        finding = {
            **payload,
            "severity": severity,
            "kind": kind,
        }
        result[f"{severity}s"].append(finding)

    basename_groups = [
        records for records in by_basename.values() if len(records) > 1
    ]
    result["same_basename"] = {
        "group_count": len(basename_groups),
        "record_count": sum(len(records) for records in basename_groups),
        "severity": policy.same_basename,
        "note": "Filename equality alone is not treated as sample identity",
    }
    return result


def _split_report(
    frames: list[FrameRecord], findings: ScanFindings
) -> dict:
    dimensions = Counter(
        (frame.width, frame.height, frame.channels) for frame in frames
    )
    videos = summarize_videos(frames)
    pair_grid_counts: Counter[tuple[str, int, int]] = Counter()
    for video in videos:
        for delta in (1, 2, 3):
            pair_grid_counts[(video.game, video.label, delta)] += getattr(
                video, f"valid_pair_count_delta{delta}"
            )
    games = sorted({frame.game for frame in frames})
    return {
        "frame_count": len(frames),
        "video_count": len(videos),
        "game_count": len(games),
        "label_counts": dict(
            sorted(Counter(frame.label for frame in frames).items())
        ),
        "dimensions": [
            {
                "width": width,
                "height": height,
                "channels": channels,
                "count": count,
            }
            for (width, height, channels), count in sorted(
                dimensions.items()
            )
        ],
        "findings": findings.to_dict(),
        "games_missing_labels": {
            game: sorted(
                {0, 1}
                - {
                    frame.label
                    for frame in frames
                    if frame.game == game
                }
            )
            for game in games
            if {
                frame.label for frame in frames if frame.game == game
            }
            != {0, 1}
        },
        "valid_pairs": {
            str(delta): sum(
                getattr(video, f"valid_pair_count_delta{delta}")
                for video in videos
            )
            for delta in (1, 2, 3)
        },
        "valid_pairs_by_game_label_delta": [
            {
                "game": game,
                "label": label,
                "delta": delta,
                "count": count,
            }
            for (game, label, delta), count in sorted(
                pair_grid_counts.items()
            )
        ],
    }


def make_audit(
    frames_by_split: dict[str, list[FrameRecord]],
    findings_by_split: dict[str, ScanFindings],
    image_spec: ImageSpec,
    duplicate_policy: DuplicatePolicy,
) -> dict:
    video_keys_by_split: dict[str, set[tuple[str, int, str]]] = {}
    source_uids_by_split: dict[str, set[str]] = {}
    for split, frames in frames_by_split.items():
        video_keys_by_split[split] = {
            (frame.game, frame.label, frame.video_id) for frame in frames
        }
        source_uids_by_split[split] = {
            source_video_uid(frame.game, frame.video_id)
            for frame in frames
        }
    split_names = [name for name in SPLIT_ORDER if name in frames_by_split]
    video_key_overlap: dict[str, list[dict]] = {}
    source_uid_overlap: dict[str, list[str]] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            pair_key = f"{left}__{right}"
            video_key_overlap[pair_key] = [
                {"game": game, "label": label, "video_id": video_id}
                for game, label, video_id in sorted(
                    video_keys_by_split[left]
                    & video_keys_by_split[right]
                )
            ]
            source_uid_overlap[pair_key] = sorted(
                source_uids_by_split[left] & source_uids_by_split[right]
            )
    return {
        "audit_format_version": AUDIT_FORMAT_VERSION,
        "expected": {
            "width": image_spec.width,
            "height": image_spec.height,
            "channels": image_spec.channels,
        },
        "policies": {
            "duplicate_policy": asdict(duplicate_policy),
        },
        "splits": {
            split: _split_report(
                frames, findings_by_split.get(split, ScanFindings())
            )
            for split, frames in frames_by_split.items()
        },
        "duplicates": analyze_content_duplicates(
            frames_by_split, duplicate_policy
        ),
        "leakage": {
            # Backwards-compatible train/test view.
            "video_keys_across_splits": video_key_overlap.get(
                "train__test", []
            ),
            "split_pair_video_key_overlap": video_key_overlap,
            "source_video_uid_overlap": source_uid_overlap,
            "video_key_check_note": (
                "source_video_uid is game::video_id; a source video must "
                "not span train/val/test"
            ),
        },
    }


def audit_warning_messages(audit: dict) -> list[str]:
    messages: list[str] = []
    for split, report in audit.get("splits", {}).items():
        warnings = report.get("findings", {}).get("warnings", [])
        if warnings:
            counts = Counter(item.get("kind", "unknown") for item in warnings)
            messages.append(
                f"{split} scan warnings: "
                + ", ".join(
                    f"{kind}={count}"
                    for kind, count in sorted(counts.items())
                )
            )
    duplicate_warnings = audit.get("duplicates", {}).get("warnings", [])
    if duplicate_warnings:
        counts = Counter(
            item.get("kind", "unknown") for item in duplicate_warnings
        )
        messages.append(
            "duplicate warnings: "
            + ", ".join(
                f"{kind}={count}"
                for kind, count in sorted(counts.items())
            )
        )
    return messages


def validate_audit(
    audit: dict,
    *,
    image_spec: ImageSpec | None = None,
    scan_policy: ScanPolicy | None = None,
    duplicate_policy: DuplicatePolicy | None = None,
    require_test_delta: int = 2,
    require_content_hash: bool = False,
    require_unique_video_keys: bool = False,
    minimum_pairs_per_game_label_delta: dict[int, int] | None = None,
) -> None:
    problems: list[str] = []
    if audit.get("audit_format_version") not in (2, AUDIT_FORMAT_VERSION):
        problems.append(
            "audit format is obsolete; rebuild indexes with tools/build_index.py"
        )
    if image_spec is not None:
        expected = audit.get("expected", {})
        actual = (
            int(expected.get("width", -1)),
            int(expected.get("height", -1)),
            int(expected.get("channels", -1)),
        )
        configured = (
            image_spec.width,
            image_spec.height,
            image_spec.channels,
        )
        if actual != configured:
            problems.append(
                f"audit image spec {actual} does not match configured "
                f"{configured}; rebuild indexes"
            )
    if duplicate_policy is not None:
        recorded_policy = (
            audit.get("policies", {}).get("duplicate_policy", {})
        )
        if recorded_policy != asdict(duplicate_policy):
            problems.append(
                "audit duplicate policy does not match configuration; "
                "rebuild indexes"
            )
    if scan_policy is not None:
        recorded_scan_policy = (
            audit.get("policies", {}).get("scan_policy", {})
        )
        if recorded_scan_policy != scan_policy.to_dict():
            problems.append(
                "audit scan policy does not match configuration; "
                "rebuild indexes"
            )

    for split in SPLIT_ORDER:
        report = audit.get("splits", {}).get(split)
        if report is None:
            if split == "train":
                problems.append(f"missing {split} audit")
            continue
        for finding in report.get("findings", {}).get("errors", []):
            problems.append(
                f"{split}: {finding.get('kind', 'error')}: "
                f"{finding.get('path', '')}"
            )
        if report.get("games_missing_labels"):
            problems.append(
                f"{split} games missing labels: "
                f"{report['games_missing_labels']}"
            )
        if report.get("frame_count", 0) == 0:
            problems.append(f"{split} has no valid frames")
        requirements = minimum_pairs_per_game_label_delta or {}
        if requirements:
            grid = {
                (
                    str(row["game"]),
                    int(row["label"]),
                    int(row["delta"]),
                ): int(row["count"])
                for row in report.get(
                    "valid_pairs_by_game_label_delta", []
                )
            }
            games = {
                str(row["game"])
                for row in report.get(
                    "valid_pairs_by_game_label_delta", []
                )
            }
            for game in sorted(games):
                for label in (0, 1):
                    for delta, minimum in sorted(requirements.items()):
                        actual = grid.get((game, label, int(delta)), 0)
                        if actual < int(minimum):
                            problems.append(
                                f"{split} {game}/label={label}/delta={delta} "
                                f"has {actual} pairs, requires {minimum}"
                            )

    test_report = audit.get("splits", {}).get("test", {})
    if int(
        test_report.get("valid_pairs", {}).get(
            str(require_test_delta), 0
        )
    ) <= 0:
        problems.append(f"test has no legal delta={require_test_delta} pairs")
    if "val" in audit.get("splits", {}):
        val_report = audit["splits"]["val"]
        if int(
            val_report.get("valid_pairs", {}).get(
                str(require_test_delta), 0
            )
        ) <= 0:
            problems.append(
                f"val has no legal delta={require_test_delta} pairs"
            )

    duplicates = audit.get("duplicates", {})
    if require_content_hash and not duplicates.get(
        "content_hash_check_enabled", False
    ):
        problems.append("content-hash duplicate check was not performed")
    for conflict in duplicates.get("errors", []):
        problems.append(
            f"duplicate conflict: {conflict.get('kind', 'error')} "
            f"sha256={conflict.get('sha256', '')}"
        )
    # Identical content crossing split boundaries is always leakage, no
    # matter which severity the duplicate policy recorded at index time.
    for warning in duplicates.get("warnings", []) + duplicates.get(
        "info", []
    ):
        if warning.get(
            "kind"
        ) == "same_label_content_overlap_across_splits":
            problems.append(
                "identical content crosses split boundaries: "
                f"sha256={warning.get('sha256', '')} "
                f"splits={warning.get('splits', [])}"
            )
    leakage = audit.get("leakage", {})
    if (
        require_unique_video_keys
        and leakage.get("video_keys_across_splits")
    ):
        problems.append(
            "train/test share video keys: "
            f"{leakage['video_keys_across_splits'][:20]}"
        )
    # Source videos must never span train/val/test. Label-independent
    # source_video_uid overlap is leakage regardless of the policy flags.
    source_overlap = leakage.get("source_video_uid_overlap")
    if isinstance(source_overlap, dict):
        for pair_key, uids in sorted(source_overlap.items()):
            if uids:
                problems.append(
                    f"source videos span the {pair_key.replace('__', '/')} "
                    f"splits: {list(uids)[:20]}"
                )
    elif leakage.get("video_keys_across_splits"):
        problems.append(
            "train/test share source video keys: "
            f"{leakage['video_keys_across_splits'][:20]}"
        )
    if problems:
        raise RuntimeError(
            "Strict dataset audit failed: " + "; ".join(problems[:100])
        )


def validate_audit_file(
    path: str | Path,
    *,
    image_spec: ImageSpec | None = None,
    scan_policy: ScanPolicy | None = None,
    duplicate_policy: DuplicatePolicy | None = None,
    require_test_delta: int = 2,
    require_content_hash: bool = False,
    require_unique_video_keys: bool = False,
    minimum_pairs_per_game_label_delta: dict[int, int] | None = None,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Strict audit is enabled but audit report is missing: {path}"
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    validate_audit(
        audit,
        image_spec=image_spec,
        scan_policy=scan_policy,
        duplicate_policy=duplicate_policy,
        require_test_delta=require_test_delta,
        require_content_hash=require_content_hash,
        require_unique_video_keys=require_unique_video_keys,
        minimum_pairs_per_game_label_delta=minimum_pairs_per_game_label_delta,
    )
    return audit


def write_index_bundle(
    train_root: str | Path,
    test_root: str | Path,
    output_dir: str | Path,
    image_spec: ImageSpec,
    scan_policy: ScanPolicy,
    duplicate_policy: DuplicatePolicy,
    *,
    val_root: str | Path | None = None,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    compute_content_hash: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    frames_by_split: dict[str, list[FrameRecord]] = {}
    findings_by_split: dict[str, ScanFindings] = {}
    roots: list[tuple[str, str | Path]] = [("train", train_root)]
    if val_root is not None:
        roots.append(("val", val_root))
    roots.append(("test", test_root))
    for split, root in roots:
        result = scan_split(
            root,
            split,
            image_spec,
            filename_pattern=filename_pattern,
            scan_policy=scan_policy,
            compute_content_hash=compute_content_hash,
        )
        frames_by_split[split] = result.frames
        findings_by_split[split] = result.findings
        write_parquet(
            result.frames, output_dir / f"{split}_frames.parquet"
        )
        write_parquet(
            summarize_videos(result.frames),
            output_dir / f"{split}_videos.parquet",
        )
        from .video_index import (
            build_video_entries,
            write_video_entries_parquet,
        )

        write_video_entries_parquet(
            build_video_entries(result.frames),
            output_dir / f"{split}_video_entries.parquet",
        )
    audit = make_audit(
        frames_by_split,
        findings_by_split,
        image_spec,
        duplicate_policy,
    )
    audit["policies"]["scan_policy"] = scan_policy.to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit


def write_split_bundle(
    train_all_root: str | Path,
    test_root: str | Path,
    output_dir: str | Path,
    image_spec: ImageSpec,
    scan_policy: ScanPolicy,
    duplicate_policy: DuplicatePolicy,
    *,
    split_config: dict,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    compute_content_hash: bool = True,
) -> dict:
    """Scan train_all once, auto-split it into train/val by source video,
    combine with the independently scanned test set, and write the full
    per-split index triplet plus the split manifest, summary and audit."""
    if split_config.get("mode") != "from_train":
        raise ValueError(
            "data.split.mode must be 'from_train' when writing a split "
            f"bundle, got {split_config.get('mode')!r}"
        )
    if (
        split_config.get("group_key", "source_video_uid")
        != "source_video_uid"
    ):
        raise ValueError("data.split.group_key must be 'source_video_uid'")
    if (
        split_config.get("balance_by", "legal_pair_count")
        != "legal_pair_count"
    ):
        raise ValueError(
            "data.split.balance_by must be 'legal_pair_count'"
        )
    output_dir = Path(output_dir)
    train_all = scan_split(
        train_all_root,
        "train",
        image_spec,
        filename_pattern=filename_pattern,
        scan_policy=scan_policy,
        compute_content_hash=compute_content_hash,
    )
    test = scan_split(
        test_root,
        "test",
        image_spec,
        filename_pattern=filename_pattern,
        scan_policy=scan_policy,
        compute_content_hash=compute_content_hash,
    )

    val_ratio = split_config["val_ratio"]
    seed = split_config["seed"]
    target_delta = int(split_config.get("target_delta", 2))
    on_new_groups = split_config.get("on_new_groups", "error")
    small_stratum_policy = split_config.get("small_stratum_policy", "error")
    manifest_path = Path(
        split_config.get("manifest", "indexes/split_manifest.parquet")
    )
    if not manifest_path.is_absolute():
        manifest_path = output_dir / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    assignment, summary = resolve_split(
        train_all.frames,
        val_ratio=val_ratio,
        seed=seed,
        target_delta=target_delta,
        manifest_path=manifest_path,
        on_new_groups=on_new_groups,
        small_stratum_policy=small_stratum_policy,
    )

    train_frames: list[FrameRecord] = []
    val_frames: list[FrameRecord] = []
    for frame in train_all.frames:
        uid = source_video_uid(frame.game, frame.video_id)
        if assignment[uid] == "val":
            val_frames.append(
                replace(
                    frame,
                    split="val",
                    sample_id=(
                        f"val:{frame.game}:{frame.label}:{frame.video_id}:"
                        f"{frame.frame_id:05d}"
                    ),
                )
            )
        else:
            train_frames.append(frame)

    frames_by_split: dict[str, list[FrameRecord]] = {
        "train": train_frames,
        "val": val_frames,
        "test": test.frames,
    }
    findings_by_split: dict[str, ScanFindings] = {
        # One physical scan covers both train and val; the same findings
        # legitimately apply to the two logical splits.
        "train": train_all.findings,
        "val": train_all.findings,
        "test": test.findings,
    }

    for split, frames in frames_by_split.items():
        write_parquet(frames, output_dir / f"{split}_frames.parquet")
        write_parquet(
            summarize_videos(frames),
            output_dir / f"{split}_videos.parquet",
        )
        from .video_index import (
            build_video_entries,
            write_video_entries_parquet,
        )

        write_video_entries_parquet(
            build_video_entries(frames),
            output_dir / f"{split}_video_entries.parquet",
        )

    audit = make_audit(
        frames_by_split,
        findings_by_split,
        image_spec,
        duplicate_policy,
    )
    audit["policies"]["scan_policy"] = scan_policy.to_dict()
    write_split_summary(summary, output_dir / "split_summary.json")
    audit["split"] = {**summary, "manifest": str(manifest_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit
