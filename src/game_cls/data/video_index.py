from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .records import FrameRecord


@dataclass(frozen=True)
class VideoEntry:
    game: str
    label: int
    video_id: str
    frame_ids: np.ndarray
    frame_paths: tuple[str, ...]
    valid_start_positions: dict[int, np.ndarray]

    def pair_paths(self, delta: int, start_position: int) -> tuple[int, int, str, str]:
        frame0_id = int(self.frame_ids[start_position])
        frame1_id = frame0_id + delta
        target = int(np.searchsorted(self.frame_ids, frame1_id))
        if target >= len(self.frame_ids) or int(self.frame_ids[target]) != frame1_id:
            raise IndexError(
                f"Invalid pair request: {self.game}/{self.video_id} "
                f"{frame0_id}+{delta}"
            )
        return (
            frame0_id,
            frame1_id,
            self.frame_paths[start_position],
            self.frame_paths[target],
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
        frame_paths = tuple(item.path for item in ordered)
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
        entries.append(
            VideoEntry(
                game=game,
                label=label,
                video_id=video_id,
                frame_ids=frame_ids,
                frame_paths=frame_paths,
                valid_start_positions=valid,
            )
        )
    return entries


def read_video_entries_parquet(
    path: str | Path, deltas: Iterable[int] = (1, 2, 3)
) -> list[VideoEntry]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Reading Parquet indexes requires pyarrow") from exc
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
    ]
    table = pq.read_table(path, columns=columns)
    return build_video_entries(
        (FrameRecord(**row) for row in table.to_pylist()),
        deltas=deltas,
    )


def video_index_memory_bytes(entries: Iterable[VideoEntry]) -> int:
    return sum(
        entry.frame_ids.nbytes
        + sum(array.nbytes for array in entry.valid_start_positions.values())
        + sum(len(path.encode("utf-8")) for path in entry.frame_paths)
        for entry in entries
    )
