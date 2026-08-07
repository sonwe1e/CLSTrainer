from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .records import FrameRecord


def video_content_id(frame_count: int, ordered_content_hashes: list[str]) -> str | None:
    """Deterministic content signature of a video (audit P0-5).

    The signature hashes the frame count plus the per-frame content SHA-256s in
    frame order, so two videos with coincidentally equal local numbering but
    different pixels get distinct canonical uids even though the source
    identity namespaces stay audit-boundary-only. Returns ``None`` when any
    frame lacks a content hash, in which case the caller falls back to the
    name-based uid.
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
    # The canonical, content-anchored source-video identity, computed once at
    # indexing time and persisted (audit P0-5). Every consumer (sidecar join,
    # hard-negative mining, subtype eval, error report) reads this instead of
    # re-deriving ``game::video_id``, so coincidentally equal train/test local
    # numbering can never collapse two distinct videos into one key. Empty when
    # read from a legacy parquet that predates the column.
    canonical_source_video_uid: str = ""
    # Optional per-video metadata joined from the sidecar AFTER split
    # (step5 P2). Never part of split/dedup identity; defaults keep the
    # legacy behavior byte-identical when no sidecar is configured.
    negative_subtype: str | None = None
    sample_weight: float = 1.0

    @property
    def source_video_uid(self) -> str:
        # The persisted canonical identity; the name-based fallback keeps the
        # label so every consumer sees one consistent format.
        return self.canonical_source_video_uid or (
            f"{self.game}::{self.label}::{self.video_id}"
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
    frames: Iterable[FrameRecord], deltas: Iterable[int] = (1, 2, 3)
) -> list[VideoEntry]:
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
        # Audit P0-5: compute the content-anchored canonical uid here, once,
        # from the ordered per-frame content hashes. The content signature
        # distinguishes coincidentally equal numbering across distinct pools
        # without touching the audit-only namespace dimension.
        content_id = video_content_id(
            len(ordered), [item.content_sha256 for item in ordered]
        )
        canonical_uid = (
            f"{game}::{int(label)}::{video_id}#{content_id}"
            if content_id is not None
            else f"{game}::{int(label)}::{video_id}"
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
                canonical_source_video_uid=canonical_uid,
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
    schema_names = set(pq.read_schema(path).names)
    if "frame_ids" in schema_names:
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
                entries.append(
                    VideoEntry(
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
                        canonical_source_video_uid=row.get(
                            "canonical_source_video_uid"
                        )
                        or "",
                        negative_subtype=row.get("negative_subtype"),
                        sample_weight=float(row.get("sample_weight", 1.0)),
                    )
                )
        return entries
    columns = [
        "sample_id",
        "split",
        "game",
        "label",
        "video_id",
        "frame_id",
        "path",
        "width",
        "height",
        "channels",
        "file_size",
        "content_sha256",
    ]
    table = pq.read_table(
        path, columns=[column for column in columns if column in schema_names]
    )
    return build_video_entries(
        (FrameRecord(**row) for row in table.to_pylist()),
        deltas=deltas,
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
            "canonical_source_video_uid": entry.canonical_source_video_uid or None,
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
        pa.Table.from_pylist(rows),
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
        + len(entry.canonical_source_video_uid.encode("utf-8"))
        for entry in entries
    )
