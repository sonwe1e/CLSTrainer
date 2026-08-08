from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from game_cls.contract import require_parquet_contract, stamp_parquet_table

from .records import FrameRecord


def video_content_id(frame_count: int, ordered_content_hashes: list[str]) -> str | None:
    """Deterministic content-version signature of a video.

    Each input token carries label, frame id and content SHA-256. Returns
    ``None`` when any frame lacks a content hash.
    """
    if not ordered_content_hashes or any(not h for h in ordered_content_hashes):
        return None
    payload = str(frame_count) + "|" + "|".join(ordered_content_hashes)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VideoEntry:
    game: str
    label: int
    video_id: str
    frame_ids: np.ndarray
    valid_start_positions: dict[int, np.ndarray]
    frame_paths: tuple[str, ...] = ()
    frame_locations: np.ndarray | None = None
    video_directory: str = ""
    # Stable physical identity and the independently changing content version.
    # Metadata is keyed by stable_source_id; samples/reports use source_version_id.
    stable_source_id: str = ""
    content_version_id: str = ""
    # Optional per-video metadata joined from the sidecar after splitting.
    # It is never part of split/dedup identity.
    negative_subtype: str | None = None
    sample_weight: float = 1.0

    @property
    def source_version_id(self) -> str:
        stable = self.stable_source_id or f"{self.game}::{self.label}::{self.video_id}"
        return (
            f"{stable}#{self.content_version_id}" if self.content_version_id else stable
        )

    def _reference(self, position: int) -> str | int:
        if self.frame_locations is not None:
            return int(self.frame_locations[position])
        if self.frame_paths:
            return self.frame_paths[position]
        if self.video_directory:
            frame_id = int(self.frame_ids[position])
            return str(
                Path(self.video_directory) / f"{self.video_id}{frame_id:05d}.png"
            )
        raise RuntimeError(
            f"Video entry has no frame references: {self.game}/{self.video_id}"
        )

    def pair_paths(
        self, delta: int, start_position: int
    ) -> tuple[int, int, str | int, str | int]:
        frame0_id = int(self.frame_ids[start_position])
        frame1_id = frame0_id + delta
        target = int(np.searchsorted(self.frame_ids, frame1_id))
        if target >= len(self.frame_ids) or int(self.frame_ids[target]) != frame1_id:
            raise IndexError(
                f"Invalid pair request: {self.game}/{self.video_id} {frame0_id}+{delta}"
            )
        return (
            frame0_id,
            frame1_id,
            self._reference(start_position),
            self._reference(target),
        )


def build_video_entries(
    frames: Iterable[FrameRecord],
    deltas: Iterable[int] = (1, 2, 3),
    *,
    identity_mode: str = "game_label_video",
    namespace: str | None = None,
    require_content_hash: bool = False,
) -> list[VideoEntry]:
    from .splitter import stable_source_id

    frames = list(frames)
    content_by_source: dict[str, list[FrameRecord]] = {}
    for frame in frames:
        stable = stable_source_id(
            frame.game,
            frame.video_id,
            frame.label,
            mode=identity_mode,
            namespace=namespace,
        )
        content_by_source.setdefault(stable, []).append(frame)
    content_versions: dict[str, str] = {}
    for stable, source_frames in content_by_source.items():
        ordered_source = sorted(
            source_frames, key=lambda item: (int(item.label), int(item.frame_id))
        )
        content_tokens = [
            (
                f"{int(item.label)}:{int(item.frame_id)}:{item.content_sha256}"
                if item.content_sha256
                else ""
            )
            for item in ordered_source
        ]
        content = video_content_id(len(ordered_source), content_tokens)
        if content is None and require_content_hash:
            raise ValueError(
                f"Source video {stable} has frames without content_sha256; "
                "a content-versioned video index cannot be built."
            )
        content_versions[stable] = content or ""
    grouped: dict[tuple[str, int, str], list[FrameRecord]] = {}
    for frame in frames:
        grouped.setdefault((frame.game, frame.label, frame.video_id), []).append(frame)
    entries = []
    for (game, label, video_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.frame_id)
        frame_ids = np.asarray([item.frame_id for item in ordered], dtype=np.int32)
        expected_names = [f"{video_id}{int(item.frame_id):05d}.png" for item in ordered]
        parents = {str(Path(item.path).parent) for item in ordered}
        compact_paths = len(parents) == 1 and all(
            Path(item.path).name == expected
            for item, expected in zip(ordered, expected_names, strict=False)
        )
        frame_paths = () if compact_paths else tuple(item.path for item in ordered)
        video_directory = next(iter(parents)) if compact_paths else ""
        id_set = set(frame_ids.tolist())
        valid = {
            int(delta): np.asarray(
                [
                    position
                    for position, frame_id in enumerate(frame_ids)
                    if int(frame_id) + int(delta) in id_set
                ],
                dtype=np.int32,
            )
            for delta in deltas
        }
        stable = stable_source_id(
            game,
            video_id,
            label,
            mode=identity_mode,
            namespace=namespace,
        )
        entries.append(
            VideoEntry(
                game=game,
                label=label,
                video_id=video_id,
                frame_ids=frame_ids,
                valid_start_positions=valid,
                frame_paths=frame_paths,
                video_directory=video_directory,
                stable_source_id=stable,
                content_version_id=content_versions[stable],
            )
        )
    return entries


def read_video_entries_parquet(
    path: str | Path, deltas: Iterable[int] = (1, 2, 3)
) -> list[VideoEntry]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Video-level index is missing: {path}. "
            "Generate the train/val/test split indexes first, e.g. "
            "'python tools/build_index.py --config configs/<profile>.yaml "
            "--train-root <train> --val-root <val> --test-root <test> "
            "--output-dir <index-dir>'. A production config with "
            "data.val_index / data.val_video_index requires a real "
            "validation split (--val-root); without one there is no "
            "independent test set either."
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet indexes requires pyarrow") from exc
    require_parquet_contract(path)
    schema_names = set(pq.read_schema(path).names)
    if "frame_ids" in schema_names:
        identity_columns = {
            "stable_source_id",
            "content_version_id",
            "source_version_id",
        }
        missing_identity = identity_columns - schema_names
        if missing_identity:
            raise ValueError(
                f"Video index {path} is missing {sorted(missing_identity)}. "
                "Rebuild it with CLSTrainer 5.0.0."
            )
        entries = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                valid = {
                    delta: np.asarray(
                        row.get(f"valid_starts_delta{delta}", []),
                        dtype=np.int32,
                    )
                    for delta in deltas
                }
                entry = VideoEntry(
                    game=row["game"],
                    label=int(row["label"]),
                    video_id=row["video_id"],
                    frame_ids=np.asarray(row["frame_ids"], dtype=np.int32),
                    valid_start_positions=valid,
                    frame_paths=tuple(row.get("frame_paths") or ()),
                    frame_locations=(
                        np.asarray(row["frame_locations"], dtype=np.int64)
                        if row.get("frame_locations") is not None
                        else None
                    ),
                    video_directory=row.get("video_directory") or "",
                    stable_source_id=row.get("stable_source_id") or "",
                    content_version_id=row.get("content_version_id") or "",
                    negative_subtype=row.get("negative_subtype"),
                    sample_weight=float(row.get("sample_weight", 1.0)),
                )
                if not entry.stable_source_id or not entry.content_version_id:
                    raise ValueError(
                        f"Video index {path} contains an unversioned identity for "
                        f"{entry.game}/{entry.label}/{entry.video_id}; rebuild it."
                    )
                if not math.isfinite(entry.sample_weight) or entry.sample_weight <= 0:
                    raise ValueError(
                        f"Video index {path} contains invalid sample_weight="
                        f"{entry.sample_weight!r} for {entry.stable_source_id}."
                    )
                recorded_source_version = row.get("source_version_id")
                if (
                    recorded_source_version
                    and recorded_source_version != entry.source_version_id
                ):
                    raise ValueError(
                        f"Video index {path} has inconsistent source_version_id for "
                        f"{entry.game}/{entry.label}/{entry.video_id}."
                    )
                entries.append(entry)
        return entries
    raise ValueError(
        f"{path} is a frame-level parquet, not a contract-5 video "
        "index. Configure the corresponding *_video_index produced by "
        "'cls-trainer dataset prepare' or 'dataset external-prepare'. "
        "Implicit frame-index fallback has been removed because it could "
        "silently change source identity semantics."
    )


def write_video_entries_parquet(
    entries: Iterable[VideoEntry], path: str | Path
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Writing video indexes requires pyarrow") from exc
    rows = []
    for entry in entries:
        row = {
            "game": entry.game,
            "label": entry.label,
            "video_id": entry.video_id,
            "frame_ids": entry.frame_ids.tolist(),
            "frame_paths": (list(entry.frame_paths) if entry.frame_paths else None),
            "frame_locations": (
                entry.frame_locations.tolist()
                if entry.frame_locations is not None
                else None
            ),
            "video_directory": entry.video_directory or None,
            "stable_source_id": entry.stable_source_id or None,
            "content_version_id": entry.content_version_id or None,
            "source_version_id": entry.source_version_id,
            "negative_subtype": entry.negative_subtype,
            "sample_weight": entry.sample_weight,
        }
        for delta in (1, 2, 3):
            row[f"valid_starts_delta{delta}"] = entry.valid_start_positions.get(
                delta, np.empty(0, dtype=np.int32)
            ).tolist()
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        stamp_parquet_table(pa.Table.from_pylist(rows)),
        path,
        compression="zstd",
    )


def video_index_memory_bytes(entries: Iterable[VideoEntry]) -> int:
    return sum(
        entry.frame_ids.nbytes
        + sum(array.nbytes for array in entry.valid_start_positions.values())
        + (entry.frame_locations.nbytes if entry.frame_locations is not None else 0)
        + sum(len(path.encode("utf-8")) for path in entry.frame_paths)
        + len(entry.video_directory.encode("utf-8"))
        + len(entry.stable_source_id.encode("utf-8"))
        + len(entry.content_version_id.encode("utf-8"))
        for entry in entries
    )
