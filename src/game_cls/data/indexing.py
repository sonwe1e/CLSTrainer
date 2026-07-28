from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from .records import (
    DEFAULT_FILENAME_PATTERN,
    FrameRecord,
    VideoRecord,
    parse_filename,
    read_png_metadata,
    summarize_videos,
)


def scan_split(
    root: str | Path,
    split: str,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
    expected_width: int = 208,
    expected_height: int = 448,
    expected_channels: int = 3,
) -> tuple[list[FrameRecord], list[dict]]:
    root = Path(root).resolve()
    frames: list[FrameRecord] = []
    issues: list[dict] = []
    if not root.is_dir():
        raise FileNotFoundError(f"{split} root does not exist: {root}")

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            issues.append(
                {
                    "path": str(path),
                    "kind": "invalid_layout",
                    "error": "expected <game>/<0|1>/<frame>.png",
                }
            )
            continue
        game, label_text, _ = relative.parts
        if label_text not in {"0", "1"}:
            issues.append({
                "path": str(path),
                "kind": "invalid_label",
                "error": "label directory must be 0 or 1",
            })
            continue
        try:
            video_id, frame_id = parse_filename(path.name, filename_pattern)
            width, height, channels = read_png_metadata(path)
        except (OSError, ValueError) as exc:
            issues.append(
                {"path": str(path), "kind": "invalid_file", "error": str(exc)}
            )
            continue
        if (width, height, channels) != (
            expected_width,
            expected_height,
            expected_channels,
        ):
            issues.append(
                {
                    "path": str(path),
                    "kind": "unexpected_dimensions",
                    "error": (
                        f"expected {expected_width}x{expected_height}x"
                        f"{expected_channels}, got {width}x{height}x{channels}"
                    ),
                }
            )
            continue
        frames.append(
            FrameRecord(
                sample_id=f"{split}:{game}:{label_text}:{video_id}:{frame_id:05d}",
                split=split,
                game=game,
                label=int(label_text),
                video_id=video_id,
                frame_id=frame_id,
                path=str(path),
                width=width,
                height=height,
                channels=channels,
                file_size=path.stat().st_size,
            )
        )
    return frames, issues


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Writing Parquet indexes requires pyarrow: python -m pip install pyarrow"
        ) from exc
    return pa, pq


def write_parquet(records: Iterable[FrameRecord | VideoRecord], path: str | Path) -> None:
    pa, pq = _pyarrow()
    rows = [asdict(record) for record in records]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def read_frame_parquet(path: str | Path) -> list[FrameRecord]:
    _, pq = _pyarrow()
    return [FrameRecord(**row) for row in pq.read_table(path).to_pylist()]


def make_audit(
    frames_by_split: dict[str, list[FrameRecord]],
    issues_by_split: dict[str, list[dict]],
    expected_width: int = 208,
    expected_height: int = 448,
    expected_channels: int = 3,
) -> dict:
    split_reports = {}
    for split, frames in frames_by_split.items():
        dimensions = Counter((f.width, f.height, f.channels) for f in frames)
        videos = summarize_videos(frames)
        split_reports[split] = {
            "frame_count": len(frames),
            "video_count": len(videos),
            "game_count": len({f.game for f in frames}),
            "label_counts": dict(sorted(Counter(f.label for f in frames).items())),
            "dimensions": [
                {"width": w, "height": h, "channels": c, "count": count}
                for (w, h, c), count in sorted(dimensions.items())
            ],
            "unexpected_dimension_count": sum(
                issue.get("kind") == "unexpected_dimensions"
                for issue in issues_by_split.get(split, [])
            ),
            "parse_or_file_issues": issues_by_split.get(split, []),
            "games_missing_labels": {
                game: sorted({0, 1} - {frame.label for frame in frames if frame.game == game})
                for game in sorted({frame.game for frame in frames})
                if {frame.label for frame in frames if frame.game == game} != {0, 1}
            },
            "valid_pairs": {
                str(delta): sum(
                    getattr(video, f"valid_pair_count_delta{delta}") for video in videos
                )
                for delta in (1, 2, 3)
            },
        }
    return {"expected": {
        "width": expected_width,
        "height": expected_height,
        "channels": expected_channels,
    }, "splits": split_reports}


def validate_audit(
    audit: dict,
    *,
    require_test_delta: int = 2,
) -> None:
    problems: list[str] = []
    for split in ("train", "test"):
        report = audit.get("splits", {}).get(split)
        if report is None:
            problems.append(f"missing {split} audit")
            continue
        if report.get("unexpected_dimension_count", 0):
            problems.append(
                f"{split} has {report['unexpected_dimension_count']} invalid dimensions"
            )
        if report.get("parse_or_file_issues"):
            problems.append(
                f"{split} has {len(report['parse_or_file_issues'])} invalid files"
            )
        if report.get("games_missing_labels"):
            problems.append(f"{split} games missing labels: {report['games_missing_labels']}")
        if report.get("frame_count", 0) == 0:
            problems.append(f"{split} has no valid frames")
    test_report = audit.get("splits", {}).get("test", {})
    if int(test_report.get("valid_pairs", {}).get(str(require_test_delta), 0)) <= 0:
        problems.append(f"test has no legal delta={require_test_delta} pairs")
    if problems:
        raise RuntimeError("Strict dataset audit failed: " + "; ".join(problems))


def validate_audit_file(path: str | Path, *, require_test_delta: int = 2) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Strict audit is enabled but audit report is missing: {path}"
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    validate_audit(audit, require_test_delta=require_test_delta)
    return audit


def write_index_bundle(
    train_root: str | Path,
    test_root: str | Path,
    output_dir: str | Path,
    filename_pattern: str = DEFAULT_FILENAME_PATTERN,
) -> dict:
    output_dir = Path(output_dir)
    frames_by_split: dict[str, list[FrameRecord]] = {}
    issues_by_split: dict[str, list[dict]] = {}
    for split, root in (("train", train_root), ("test", test_root)):
        frames, issues = scan_split(root, split, filename_pattern)
        frames_by_split[split] = frames
        issues_by_split[split] = issues
        write_parquet(frames, output_dir / f"{split}_frames.parquet")
        write_parquet(summarize_videos(frames), output_dir / f"{split}_videos.parquet")
    audit = make_audit(frames_by_split, issues_by_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit
