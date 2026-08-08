from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

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
from .splitter import resolve_split, source_video_uid, write_split_summary

AUDIT_FORMAT_VERSION = 3

# Ordered split roles of the train/validation/test protocol.
SPLIT_ORDER = ("train", "val", "test")


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


# ---------------------------------------------------------------------------
# Split bundle transaction (audit P0-9).
# ---------------------------------------------------------------------------

BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"
BUNDLE_MANIFEST_VERSION = 1

# Every artifact a complete split bundle publishes, relative to the index dir.
# Order is the publish order; the manifest is deliberately NOT in this list
# because it is the commit record written after all of them.
BUNDLE_ARTIFACT_NAMES: tuple[str, ...] = (
    "train_frames.parquet",
    "val_frames.parquet",
    "test_frames.parquet",
    "train_videos.parquet",
    "val_videos.parquet",
    "test_videos.parquet",
    "train_video_entries.parquet",
    "val_video_entries.parquet",
    "test_video_entries.parquet",
    "split_summary.json",
    "audit.json",
)


class SplitBundleError(RuntimeError):
    """A split bundle is unsealed, incomplete, corrupt, or mixed-generation."""


def _bundle_manifest_payload(
    index_dir: Path, bundle_id: str, artifact_names: Iterable[str]
) -> dict:
    import datetime

    artifacts = {}
    for name in artifact_names:
        path = index_dir / name
        if path.is_file():
            artifacts[name] = _sha256(path)
    return {
        "format_version": BUNDLE_MANIFEST_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.datetime.now(datetime.UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "artifacts": artifacts,
    }


def write_bundle_manifest(
    index_dir: str | Path,
    bundle_id: str,
    *,
    artifact_names: Iterable[str] | None = None,
) -> Path:
    """Write ``bundle_manifest.json`` atomically -- the bundle's commit record.

    This is written LAST, after every artifact is in place, and via a temp file
    plus ``os.replace`` so it either exists complete or not at all. That
    ordering is what makes a mixed-generation bundle detectable: a crash partway
    through publishing leaves the PREVIOUS manifest (or none), whose recorded
    hashes no longer match the files on disk, so ``verify_split_bundle`` refuses
    instead of training on a train=NEW/test=OLD mixture in which every file
    exists (audit P0-9).
    """
    import os

    index_dir = Path(index_dir)
    payload = _bundle_manifest_payload(
        index_dir, bundle_id, artifact_names or BUNDLE_ARTIFACT_NAMES
    )
    target = index_dir / BUNDLE_MANIFEST_FILENAME
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)
    return target


def read_bundle_manifest(index_dir: str | Path) -> dict | None:
    """The bundle manifest, or ``None`` when the directory is unsealed."""
    path = Path(index_dir) / BUNDLE_MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SplitBundleError(
            f"Bundle manifest is unreadable: {path}: {exc}. Re-run "
            "'cls-trainer dataset prepare' or 'cls-trainer dataset seal'."
        ) from exc
    return payload if isinstance(payload, dict) else None


def _stamped_bundle_id(path: Path) -> str | None:
    """``bundle_id`` recorded inside a JSON artifact, when present."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("bundle_id")
    return str(value) if value else None


def verify_split_bundle(
    index_dir: str | Path, *, require_manifest: bool = True
) -> dict | None:
    """Refuse a bundle that is not provably one generation (audit P0-9).

    ``write_split_bundle`` used to overwrite artifacts in place, in order:
    train, val, test, video indexes, summary, audit. A crash partway through,
    over a directory that already held a complete older bundle, left
    ``train=NEW, test=OLD, audit=OLD`` with *every file present* -- so an
    existence check reported nothing missing and training proceeded on a
    silently mixed generation.

    Existence checks cannot detect that, because existence is exactly what the
    mixed state satisfies. This verifies the commit record instead:

    * the manifest exists (the bundle was published, not interrupted),
    * every artifact it lists is present and hashes to the recorded value,
    * ``audit.json`` and ``split_summary.json`` agree on ``bundle_id``.

    Each failure gets its own message because each implies a different operator
    action: seal an unsealed legacy directory, re-prepare a mixed one,
    re-prepare or restore a corrupt artifact.
    """
    index_dir = Path(index_dir)
    manifest = read_bundle_manifest(index_dir)
    if manifest is None:
        if not require_manifest:
            return None
        raise SplitBundleError(
            f"Index bundle {index_dir} has no {BUNDLE_MANIFEST_FILENAME}, so a "
            "mixed-generation bundle (e.g. train from a new prepare, test left "
            "from an older one) cannot be ruled out -- every file can be "
            "present and still disagree. Either rebuild the bundle:\n"
            "    cls-trainer dataset prepare --config <config> "
            "--train-root <train_all> --test-root <test>\n"
            "or, if you know this directory is one consistent generation, "
            "adopt it:\n"
            "    cls-trainer dataset seal --config <config>"
        )
    version = int(manifest.get("format_version", 0))
    if version != BUNDLE_MANIFEST_VERSION:
        raise SplitBundleError(
            f"Unsupported bundle manifest version {version} in {index_dir} "
            f"(expected {BUNDLE_MANIFEST_VERSION}). Re-run "
            "'cls-trainer dataset prepare'."
        )
    bundle_id = str(manifest.get("bundle_id") or "")
    if not bundle_id:
        raise SplitBundleError(
            f"Bundle manifest in {index_dir} records no bundle_id, so its "
            "artifacts cannot be tied to one generation. Re-run "
            "'cls-trainer dataset prepare' or 'cls-trainer dataset seal'."
        )
    artifacts = manifest.get("artifacts") or {}
    if not artifacts:
        raise SplitBundleError(
            f"Bundle manifest in {index_dir} lists no artifacts; it cannot "
            "prove the bundle is intact. Re-run 'cls-trainer dataset seal'."
        )
    missing: list[str] = []
    corrupt: list[str] = []
    for name, expected in sorted(artifacts.items()):
        path = index_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        if _sha256(path) != str(expected):
            corrupt.append(name)
    if missing:
        raise SplitBundleError(
            f"Index bundle {index_dir} is incomplete: the manifest lists "
            f"{', '.join(missing)}, which is missing from disk. The bundle was "
            "partially deleted or partially copied; re-run "
            "'cls-trainer dataset prepare'."
        )
    if corrupt:
        raise SplitBundleError(
            f"Index bundle {index_dir} is a MIXED GENERATION or corrupt: "
            f"{', '.join(corrupt)} does not match the hash recorded when the "
            "bundle was committed. Some artifacts come from a different "
            "prepare than the rest, so train/val/test are not one consistent "
            "split. Re-run 'cls-trainer dataset prepare' to rebuild the whole "
            "bundle."
        )
    for name in ("audit.json", "split_summary.json"):
        stamped = _stamped_bundle_id(index_dir / name)
        if stamped is not None and stamped != bundle_id:
            raise SplitBundleError(
                f"Index bundle {index_dir} is a MIXED GENERATION: {name} "
                f"carries bundle_id {stamped} but the manifest commits "
                f"{bundle_id}. This file came from a different prepare; "
                "re-run 'cls-trainer dataset prepare'."
            )
    return manifest


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
                findings.add_ignored("non_directory_at_label_level", label_dir)
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
    findings = ScanFindings(ignored_example_limit=scan_policy.ignored_example_limit)
    frames: list[FrameRecord] = []
    if not root.is_dir():
        raise FileNotFoundError(f"{split} root does not exist: {root}")

    for game, label, path in iter_frame_candidates(root, scan_policy, findings):
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
            content_sha256 = _sha256(path) if compute_content_hash else ""
        except OSError as exc:
            findings.add("error", "unreadable_file", path, str(exc))
            continue
        label_text = str(label)
        frames.append(
            FrameRecord(
                sample_id=(f"{split}:{game}:{label_text}:{video_id}:{frame_id:05d}"),
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
            "Writing Parquet indexes requires pyarrow: python -m pip install pyarrow"
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
    return [FrameRecord(**row) for row in pq.read_table(path).to_pylist()]


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

    result: dict[str, Any] = {
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

    basename_groups = [records for records in by_basename.values() if len(records) > 1]
    result["same_basename"] = {
        "group_count": len(basename_groups),
        "record_count": sum(len(records) for records in basename_groups),
        "severity": policy.same_basename,
        "note": "Filename equality alone is not treated as sample identity",
    }
    return result


def _split_report(frames: list[FrameRecord], findings: ScanFindings) -> dict:
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
        "label_counts": dict(sorted(Counter(frame.label for frame in frames).items())),
        "dimensions": [
            {
                "width": width,
                "height": height,
                "channels": channels,
                "count": count,
            }
            for (width, height, channels), count in sorted(dimensions.items())
        ],
        "findings": findings.to_dict(),
        "games_missing_labels": {
            game: sorted(
                {0, 1} - {frame.label for frame in frames if frame.game == game}
            )
            for game in games
            if {frame.label for frame in frames if frame.game == game} != {0, 1}
        },
        "valid_pairs": {
            str(delta): sum(
                getattr(video, f"valid_pair_count_delta{delta}") for video in videos
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
            for (game, label, delta), count in sorted(pair_grid_counts.items())
        ],
    }


def _video_key(frame: FrameRecord, namespace: str | None) -> tuple:
    """Cross-split video-key identity for one frame.

    Under a source namespace the key is the 4-tuple ``(namespace, game,
    label, video_id)`` so coincidentally equal local numbering in a
    distinct source pool does not intersect; otherwise the legacy 3-tuple
    keeps the default path byte-for-byte identical.
    """
    if namespace is not None:
        return (namespace, frame.game, frame.label, frame.video_id)
    return (frame.game, frame.label, frame.video_id)


def _video_key_entry(item: tuple) -> dict:
    """Render a cross-split video-key overlap entry for storage.

    Namespaced keys (4-tuples) include a ``namespace`` field; the legacy
    3-tuple keeps the ``{"game", "label", "video_id"}`` shape so
    pre-namespace audit consumers are unchanged.
    """
    if len(item) == 4:
        namespace, game, label, video_id = item
        return {
            "game": game,
            "label": label,
            "video_id": video_id,
            "namespace": namespace,
        }
    game, label, video_id = item
    return {"game": game, "label": label, "video_id": video_id}


def _source_uid_content_classification(
    frames_by_split: dict[str, list[FrameRecord]],
    source_uids_by_split: dict[str, set[str]],
    namespaces_by_split: dict[str, str],
    identity_mode: str,
) -> dict:
    """Per-colliding-uid diagnostic: how much frame content is shared.

    For every split pair with non-empty source-uid overlap, report for each
    colliding uid how many frames share identical content (SHA-256) across
    the two splits. This is a *diagnostic* to help an operator tell an ID
    numbering collision (shared=0) apart from a real source-video leak
    (shared>0); it never relaxes the strict gate. ``content_hashes_available``
    is False when any involved frame lacks a content hash, in which case the
    counts are best-effort.
    """
    grouped: dict[str, dict[str, list[FrameRecord]]] = {}
    hashes_available = True
    for split, frames in frames_by_split.items():
        namespace = namespaces_by_split.get(split)
        for frame in frames:
            # Audit B2: the classification key must use the SAME identity mode
            # as the strict overlap set (source_uids_by_split), or a
            # ``game_label_video`` audit would group label-distinct videos into
            # one ``game::video`` bucket and emit a misleading diagnostic.
            uid = source_video_uid(
                frame.game,
                frame.video_id,
                frame.label,
                mode=identity_mode,
                namespace=namespace,
            )
            grouped.setdefault(split, {}).setdefault(uid, []).append(frame)
            if not frame.content_sha256:
                hashes_available = False
    split_names = [name for name in SPLIT_ORDER if name in frames_by_split]
    pairs: dict[str, list[dict]] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            pair_key = f"{left}__{right}"
            overlap = sorted(source_uids_by_split[left] & source_uids_by_split[right])
            if not overlap:
                continue
            entries: list[dict] = []
            for uid in overlap:
                shared_hashes = {
                    frame.content_sha256
                    for frame in grouped[left][uid]
                    if frame.content_sha256
                } & {
                    frame.content_sha256
                    for frame in grouped[right][uid]
                    if frame.content_sha256
                }
                shared_frames = sum(
                    1
                    for frame in grouped[left][uid] + grouped[right][uid]
                    if frame.content_sha256 in shared_hashes
                )
                entries.append(
                    {
                        "source_video_uid": uid,
                        "shared_content_frames": shared_frames,
                    }
                )
            pairs[pair_key] = entries
    return {"content_hashes_available": hashes_available, "pairs": pairs}


def format_uid_overlap_content_classification(classification: dict) -> str:
    """Human-readable collision classification for ``dataset audit``."""
    pairs = classification.get("pairs", {})
    if not pairs:
        return ""
    lines = [
        "source-video overlap content classification (diagnostic only; "
        "the strict gate stays in force):"
    ]
    if not classification.get("content_hashes_available", True):
        lines.append(
            "  (content hashes were unavailable on some frames; shared-frame "
            "counts are best-effort)"
        )
    for pair_key, entries in sorted(pairs.items()):
        for entry in entries:
            shared = entry["shared_content_frames"]
            hint = (
                "likely numbering collision"
                if shared == 0
                else "likely same video, real leakage"
            )
            lines.append(
                f"  {entry['source_video_uid']} "
                f"({pair_key.replace('__', '/')}): "
                f"shared content frames={shared} -> {hint}"
            )
    return "\n".join(lines)


def make_audit(
    frames_by_split: dict[str, list[FrameRecord]],
    findings_by_split: dict[str, ScanFindings],
    image_spec: ImageSpec,
    duplicate_policy: DuplicatePolicy,
    *,
    identity_mode: str = "game_video",
    namespaces_by_split: dict[str, str] | None = None,
) -> dict:
    namespaces_by_split = namespaces_by_split or {}
    video_keys_by_split: dict[str, set[tuple]] = {}
    source_uids_by_split: dict[str, set[str]] = {}
    for split, frames in frames_by_split.items():
        namespace = namespaces_by_split.get(split)
        video_keys_by_split[split] = {_video_key(frame, namespace) for frame in frames}
        source_uids_by_split[split] = {
            source_video_uid(
                frame.game,
                frame.video_id,
                frame.label,
                mode=identity_mode,
                namespace=namespace,
            )
            for frame in frames
        }
    split_names = [name for name in SPLIT_ORDER if name in frames_by_split]
    video_key_overlap: dict[str, list[dict]] = {}
    source_uid_overlap: dict[str, list[str]] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            pair_key = f"{left}__{right}"
            video_key_overlap[pair_key] = [
                _video_key_entry(item)
                for item in sorted(
                    video_keys_by_split[left] & video_keys_by_split[right]
                )
            ]
            source_uid_overlap[pair_key] = sorted(
                source_uids_by_split[left] & source_uids_by_split[right]
            )
    leakage: dict[str, object] = {
        # Backwards-compatible train/test view.
        "video_keys_across_splits": video_key_overlap.get("train__test", []),
        "split_pair_video_key_overlap": video_key_overlap,
        "source_video_uid_overlap": source_uid_overlap,
        "source_identity_mode": identity_mode,
        "video_key_check_note": (
            f"source_video_uid identity mode is {identity_mode}"
            + (
                f" with source namespaces {dict(sorted(namespaces_by_split.items()))}"
                if namespaces_by_split
                else ""
            )
            + "; a source video must not span train/val/test"
        ),
    }
    if namespaces_by_split:
        leakage["source_identity_namespaces"] = dict(
            sorted(namespaces_by_split.items())
        )
    classification = _source_uid_content_classification(
        frames_by_split,
        source_uids_by_split,
        namespaces_by_split,
        identity_mode,
    )
    if classification["pairs"]:
        leakage["source_uid_overlap_content_classification"] = classification
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
            split: _split_report(frames, findings_by_split.get(split, ScanFindings()))
            for split, frames in frames_by_split.items()
        },
        "duplicates": analyze_content_duplicates(frames_by_split, duplicate_policy),
        "leakage": leakage,
    }


def audit_warning_messages(audit: dict) -> list[str]:
    messages: list[str] = []
    for split, report in audit.get("splits", {}).items():
        warnings = report.get("findings", {}).get("warnings", [])
        if warnings:
            counts = Counter(item.get("kind", "unknown") for item in warnings)
            messages.append(
                f"{split} scan warnings: "
                + ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
            )
    duplicate_warnings = audit.get("duplicates", {}).get("warnings", [])
    if duplicate_warnings:
        counts = Counter(item.get("kind", "unknown") for item in duplicate_warnings)
        messages.append(
            "duplicate warnings: "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
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
    identity_mode: str = "game_video",
    namespaces_by_split: dict[str, str] | None = None,
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
        recorded_policy = audit.get("policies", {}).get("duplicate_policy", {})
        if recorded_policy != asdict(duplicate_policy):
            problems.append(
                "audit duplicate policy does not match configuration; rebuild indexes"
            )
    if scan_policy is not None:
        recorded_scan_policy = audit.get("policies", {}).get("scan_policy", {})
        if recorded_scan_policy != scan_policy.to_dict():
            problems.append(
                "audit scan policy does not match configuration; rebuild indexes"
            )

    for split in SPLIT_ORDER:
        report = audit.get("splits", {}).get(split)
        if report is None:
            if split == "train":
                problems.append(f"missing {split} audit")
            continue
        for finding in report.get("findings", {}).get("errors", []):
            problems.append(
                f"{split}: {finding.get('kind', 'error')}: {finding.get('path', '')}"
            )
        if report.get("games_missing_labels"):
            problems.append(
                f"{split} games missing labels: {report['games_missing_labels']}"
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
                for row in report.get("valid_pairs_by_game_label_delta", [])
            }
            games = {
                str(row["game"])
                for row in report.get("valid_pairs_by_game_label_delta", [])
            }
            for game in sorted(games):
                for label in (0, 1):
                    for delta, minimum in sorted(requirements.items()):
                        pair_count = grid.get((game, label, int(delta)), 0)
                        if pair_count < int(minimum):
                            problems.append(
                                f"{split} {game}/label={label}/delta={delta} "
                                f"has {pair_count} pairs, requires {minimum}"
                            )

    test_report = audit.get("splits", {}).get("test", {})
    if int(test_report.get("valid_pairs", {}).get(str(require_test_delta), 0)) <= 0:
        problems.append(f"test has no legal delta={require_test_delta} pairs")
    if "val" in audit.get("splits", {}):
        val_report = audit["splits"]["val"]
        if int(val_report.get("valid_pairs", {}).get(str(require_test_delta), 0)) <= 0:
            problems.append(f"val has no legal delta={require_test_delta} pairs")

    duplicates = audit.get("duplicates", {})
    if require_content_hash and not duplicates.get("content_hash_check_enabled", False):
        problems.append("content-hash duplicate check was not performed")
    for conflict in duplicates.get("errors", []):
        problems.append(
            f"duplicate conflict: {conflict.get('kind', 'error')} "
            f"sha256={conflict.get('sha256', '')}"
        )
    # Identical content crossing split boundaries is always leakage, no
    # matter which severity the duplicate policy recorded at index time.
    for warning in duplicates.get("warnings", []) + duplicates.get("info", []):
        if warning.get("kind") == "same_label_content_overlap_across_splits":
            problems.append(
                "identical content crosses split boundaries: "
                f"sha256={warning.get('sha256', '')} "
                f"splits={warning.get('splits', [])}"
            )
    leakage = audit.get("leakage", {})
    if require_unique_video_keys and leakage.get("video_keys_across_splits"):
        problems.append(
            f"train/test share video keys: {leakage['video_keys_across_splits'][:20]}"
        )
    # The audit must have been built under the same source identity mode as
    # the current configuration, otherwise its leakage verdicts do not mean
    # what the caller thinks they mean.
    if leakage.get("source_identity_mode", "game_video") != identity_mode:
        problems.append(
            "audit source identity mode does not match configuration; rebuild indexes"
        )
    # The audit must also have been built under the same per-split source
    # namespaces: its overlap verdicts only mean what the caller expects if
    # the provenance boundary it was computed under is the configured one.
    if leakage.get("source_identity_namespaces", {}) != (namespaces_by_split or {}):
        problems.append(
            "audit source identity namespaces do not match configuration; "
            "rebuild indexes"
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
        raise RuntimeError("Strict dataset audit failed: " + "; ".join(problems[:100]))


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
    identity_mode: str = "game_video",
    namespaces_by_split: dict[str, str] | None = None,
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
        identity_mode=identity_mode,
        namespaces_by_split=namespaces_by_split,
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
    identity_mode: str = "game_video",
    namespaces_by_split: dict[str, str] | None = None,
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
        write_parquet(result.frames, output_dir / f"{split}_frames.parquet")
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
        identity_mode=identity_mode,
        namespaces_by_split=namespaces_by_split,
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
    identity_mode: str = "game_video",
    namespaces_by_split: dict[str, str] | None = None,
) -> dict:
    """Scan train_all once, auto-split it into train/val by source video,
    combine with the independently scanned test set, and write the full
    per-split index triplet plus the split manifest, summary and audit."""
    if split_config.get("mode") != "from_train":
        raise ValueError(
            "data.split.mode must be 'from_train' when writing a split "
            f"bundle, got {split_config.get('mode')!r}"
        )
    if split_config.get("group_key", "source_video_uid") != "source_video_uid":
        raise ValueError("data.split.group_key must be 'source_video_uid'")
    if split_config.get("balance_by", "legal_pair_count") != "legal_pair_count":
        raise ValueError("data.split.balance_by must be 'legal_pair_count'")
    if identity_mode not in ("game_video", "game_label_video"):
        raise ValueError(
            "identity_mode must be 'game_video' or 'game_label_video', "
            f"got {identity_mode!r}"
        )
    # A caller may record the mode inside split_config (tools/build_index.py
    # does); if it disagrees with the explicit kwarg, fail rather than
    # silently split under the wrong leakage unit.
    split_identity = split_config.get("source_identity_mode")
    if split_identity is not None and split_identity != identity_mode:
        raise ValueError(
            "split_config['source_identity_mode'] and the identity_mode "
            f"argument disagree: {split_identity!r} != {identity_mode!r}"
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
    manifest_path = Path(split_config.get("manifest", "indexes/split_manifest.parquet"))
    if not manifest_path.is_absolute():
        manifest_path = output_dir / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # step7 section 8: the source-identity structure is analysed before the
    # split runs. On success the report rides along on the returned audit;
    # on failure it is attached to the error so the operator sees the data
    # structure instead of a bare traceback.
    from .splitter import format_source_identity_precheck, source_identity_precheck

    source_identity_report = source_identity_precheck(
        train_all.frames, identity_mode=identity_mode
    )
    try:
        assignment, summary = resolve_split(
            train_all.frames,
            val_ratio=val_ratio,
            seed=seed,
            target_delta=target_delta,
            manifest_path=manifest_path,
            on_new_groups=on_new_groups,
            small_stratum_policy=small_stratum_policy,
            identity_mode=identity_mode,
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc}\n\n{format_source_identity_precheck(source_identity_report)}"
        ) from exc

    train_frames: list[FrameRecord] = []
    val_frames: list[FrameRecord] = []
    for frame in train_all.frames:
        uid = source_video_uid(
            frame.game, frame.video_id, frame.label, mode=identity_mode
        )
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

    # Audit P0-9: publish the bundle as a transaction. Writing artifacts
    # straight into output_dir overwrote them one at a time (train, val, test,
    # video indexes, summary, audit), so a crash partway through -- over a
    # directory that already held a complete older bundle -- left
    # train=NEW/test=OLD with every file present, which no existence check can
    # detect. Instead: stage everything, move it into place, then write the
    # commit record LAST. A crash before the commit leaves the previous manifest
    # (or none), whose hashes no longer match, so verify_split_bundle refuses.
    import os
    import shutil
    from uuid import uuid4

    from .video_index import build_video_entries, write_video_entries_parquet

    bundle_id = uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / f".staging-{bundle_id}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for split, frames in frames_by_split.items():
            write_parquet(frames, staging / f"{split}_frames.parquet")
            write_parquet(
                summarize_videos(frames),
                staging / f"{split}_videos.parquet",
            )
            write_video_entries_parquet(
                build_video_entries(frames),
                staging / f"{split}_video_entries.parquet",
            )

        audit = make_audit(
            frames_by_split,
            findings_by_split,
            image_spec,
            duplicate_policy,
            identity_mode=identity_mode,
            namespaces_by_split=namespaces_by_split,
        )
        # The source identity precheck rides along on the returned audit dict.
        audit["source_identity_precheck"] = source_identity_report
        audit["policies"]["scan_policy"] = scan_policy.to_dict()
        # Stamp the generation into both JSON artifacts, so an audit or summary
        # swapped in from another prepare is caught by ID disagreement even if
        # its recorded hash somehow matched.
        audit["bundle_id"] = bundle_id
        audit["split"] = {**summary, "manifest": str(manifest_path)}
        write_split_summary(
            {**summary, "bundle_id": bundle_id}, staging / "split_summary.json"
        )
        (staging / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Publish. Still not a single atomic step -- POSIX gives no atomic
        # multi-file rename -- but every intermediate state is now *detectable*,
        # which is the property that was missing.
        for name in BUNDLE_ARTIFACT_NAMES:
            staged = staging / name
            if staged.is_file():
                os.replace(staged, output_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # The split manifest is deliberately NOT hash-committed. resolve_split
    # writes/extends it BEFORE any parquet is staged, so a crash in that window
    # leaves a manifest that moved while the parquets did not -- and hashing it
    # would report MIXED GENERATION even though train/val/test are still one
    # internally consistent set. That is a false positive on a bundle that is
    # actually fine, so the manifest is checked for existence only (see
    # cli.dataset._split_bundle_artifacts); its assignments are already
    # protected by resolve_split's own on_new_groups contract.
    write_bundle_manifest(output_dir, bundle_id)
    return audit
