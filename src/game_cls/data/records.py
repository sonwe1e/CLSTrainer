from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import struct
from typing import Iterable


DEFAULT_FILENAME_PATTERN = r"^(?P<video_id>\d{2})(?P<frame_id>\d{5})\.png$"


@dataclass(frozen=True)
class FrameRecord:
    sample_id: str
    split: str
    game: str
    label: int
    video_id: str
    frame_id: int
    path: str
    width: int
    height: int
    channels: int
    file_size: int
    content_sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VideoRecord:
    split: str
    game: str
    label: int
    video_id: str
    frame_count: int
    min_frame_id: int
    max_frame_id: int
    valid_pair_count_delta1: int
    valid_pair_count_delta2: int
    valid_pair_count_delta3: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairSample:
    game: str
    label: int
    video_id: str
    frame0_id: int
    frame1_id: int
    delta: int
    image0_path: str
    image1_path: str


def parse_filename(
    filename: str, pattern: str = DEFAULT_FILENAME_PATTERN
) -> tuple[str, int]:
    match = re.fullmatch(pattern, filename)
    if not match:
        raise ValueError(f"Invalid frame filename: {filename}")
    return match.group("video_id"), int(match.group("frame_id"))


def read_png_metadata(path: str | Path) -> tuple[int, int, int]:
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(26)
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {path}")
    if header[12:16] != b"IHDR":
        raise ValueError(f"PNG has no leading IHDR chunk: {path}")
    width, height = struct.unpack(">II", header[16:24])
    color_type = header[25]
    channels_by_color_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_color_type:
        raise ValueError(f"Unsupported PNG color type {color_type}: {path}")
    return width, height, channels_by_color_type[color_type]


def build_pairs(frames: Iterable[FrameRecord], delta: int) -> list[PairSample]:
    if delta <= 0:
        raise ValueError("delta must be positive")
    grouped: dict[tuple[str, int, str], dict[int, FrameRecord]] = {}
    for frame in frames:
        key = (frame.game, frame.label, frame.video_id)
        grouped.setdefault(key, {})[frame.frame_id] = frame

    pairs: list[PairSample] = []
    for (game, label, video_id), by_id in sorted(grouped.items()):
        for frame0_id in sorted(by_id):
            frame1_id = frame0_id + delta
            if frame1_id not in by_id:
                continue
            frame0, frame1 = by_id[frame0_id], by_id[frame1_id]
            pairs.append(
                PairSample(
                    game=game,
                    label=label,
                    video_id=video_id,
                    frame0_id=frame0_id,
                    frame1_id=frame1_id,
                    delta=delta,
                    image0_path=frame0.path,
                    image1_path=frame1.path,
                )
            )
    return pairs


def summarize_videos(frames: Iterable[FrameRecord]) -> list[VideoRecord]:
    grouped: dict[tuple[str, str, int, str], list[FrameRecord]] = {}
    for frame in frames:
        key = (frame.split, frame.game, frame.label, frame.video_id)
        grouped.setdefault(key, []).append(frame)
    records: list[VideoRecord] = []
    for (split, game, label, video_id), group in sorted(grouped.items()):
        ids = {frame.frame_id for frame in group}
        pair_counts = {
            delta: sum(frame_id + delta in ids for frame_id in ids)
            for delta in (1, 2, 3)
        }
        records.append(
            VideoRecord(
                split=split,
                game=game,
                label=label,
                video_id=video_id,
                frame_count=len(ids),
                min_frame_id=min(ids),
                max_frame_id=max(ids),
                valid_pair_count_delta1=pair_counts[1],
                valid_pair_count_delta2=pair_counts[2],
                valid_pair_count_delta3=pair_counts[3],
            )
        )
    return records
