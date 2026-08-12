from __future__ import annotations

import bisect
import json
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .augments import PairedAugment, build_train_augment
from .data import (
    _backend_roots,
    _normalize_delta_probabilities,
    _normalize_deltas,
    _train_deltas,
    split_train_val_videos,
)


@dataclass(frozen=True)
class VideoRecord:
    game: str
    label: int
    video_id: str
    path: str
    frame_count: int
    fps_num: int
    fps_den: int
    width: int
    height: int

    @property
    def key(self) -> tuple[str, int, str]:
        return self.game, self.label, self.video_id

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    def pair_count_for_delta(self, delta: int) -> int:
        return max(0, self.frame_count - int(delta))


def _fraction(text: str) -> Fraction:
    if not text or text in {"0/0", "N/A"}:
        raise ValueError(f"Invalid video frame rate: {text!r}")
    value = Fraction(text)
    if value <= 0:
        raise ValueError(f"Invalid video frame rate: {text!r}")
    return value


def _load_manifest(root: Path) -> dict[str, dict]:
    path = root / ".transcode_manifest.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
    return entries if isinstance(entries, dict) else {}


def _probe_video(path: Path, ffprobe_bin: str) -> tuple[int, Fraction, int, int]:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    fps = _fraction(str(stream["avg_frame_rate"]))
    frames_text = str(stream.get("nb_frames", "N/A"))
    if frames_text not in {"", "N/A", "None"}:
        frame_count = int(frames_text)
    else:
        duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
        if duration <= 0:
            raise ValueError(
                f"Cannot determine frame count for {path}; use videos produced by transcode_videos.py"
            )
        frame_count = int(round(duration * float(fps)))
    return frame_count, fps, int(stream["width"]), int(stream["height"])


def scan_video_root(
    root: str | Path,
    *,
    extensions: Sequence[str],
    ffprobe_bin: str = "ffprobe",
) -> list[VideoRecord]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video root does not exist: {root}")
    allowed = {str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in extensions}
    manifest = _load_manifest(root)

    videos: list[VideoRecord] = []
    for game_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for label in (0, 1):
            label_dir = game_dir / str(label)
            if not label_dir.is_dir():
                continue
            for path in sorted(p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed):
                rel = path.relative_to(root).as_posix()
                entry = manifest.get(rel, {})
                output_meta = entry.get("output", {}) if isinstance(entry, dict) else {}
                if output_meta and all(
                    key in output_meta for key in ("frames", "fps_num", "fps_den", "width", "height")
                ):
                    frame_count = int(output_meta["frames"])
                    fps = Fraction(int(output_meta["fps_num"]), int(output_meta["fps_den"]))
                    width = int(output_meta["width"])
                    height = int(output_meta["height"])
                else:
                    frame_count, fps, width, height = _probe_video(path, ffprobe_bin)
                if frame_count <= 1:
                    continue
                video_id = path.relative_to(label_dir).with_suffix("").as_posix()
                videos.append(
                    VideoRecord(
                        game=game_dir.name,
                        label=label,
                        video_id=video_id,
                        path=str(path),
                        frame_count=frame_count,
                        fps_num=fps.numerator,
                        fps_den=fps.denominator,
                        width=width,
                        height=height,
                    )
                )
    if not videos:
        raise ValueError(f"No videos found under {root} for extensions={sorted(allowed)}")
    return videos


class _VideoChunkCache:
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        chunk_frames: int,
        cache_chunks: int,
        ffmpeg_bin: str,
        ffmpeg_threads: int,
    ) -> None:
        self.image_size = image_size
        self.chunk_frames = int(chunk_frames)
        self.cache_chunks = int(cache_chunks)
        self.ffmpeg_bin = ffmpeg_bin
        self.ffmpeg_threads = int(ffmpeg_threads)
        self._cache: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()

    def _decode(self, video: VideoRecord, start: int) -> torch.Tensor:
        count = min(self.chunk_frames, video.frame_count - start)
        if count <= 0:
            raise IndexError(start)
        height, width = self.image_size
        if (video.height, video.width) != (height, width):
            raise ValueError(
                f"Video {video.path} is {video.height}x{video.width}; expected preprocessed "
                f"{height}x{width}. Run tools/transcode_videos.py first."
            )
        start_seconds = start * video.fps_den / video.fps_num
        cmd = [
            self.ffmpeg_bin,
            "-v",
            "error",
            "-nostdin",
            "-threads",
            str(self.ffmpeg_threads),
            "-ss",
            f"{start_seconds:.9f}",
            "-i",
            video.path,
            "-map",
            "0:v:0",
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True)
        frame_bytes = height * width * 3
        raw = completed.stdout
        decoded = len(raw) // frame_bytes
        if decoded < count:
            raise RuntimeError(
                f"ffmpeg returned {decoded}/{count} frames for {video.path} at start={start}. "
                "The video may be variable-frame-rate or its manifest may be stale."
            )
        array = np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8).copy()
        array = array.reshape(count, height, width, 3)
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    def get(self, video: VideoRecord, left: int, right: int) -> tuple[torch.Tensor, torch.Tensor]:
        aligned = (left // self.chunk_frames) * self.chunk_frames
        if right >= aligned + self.chunk_frames:
            aligned = max(0, right - self.chunk_frames + 1)
        key = (video.path, aligned)
        frames = self._cache.get(key)
        if frames is None:
            frames = self._decode(video, aligned)
            self._cache[key] = frames
            while len(self._cache) > self.cache_chunks:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        local_left = left - aligned
        local_right = right - aligned
        if local_left < 0 or local_right >= len(frames):
            raise RuntimeError("Internal chunk boundary error")
        return frames[local_left], frames[local_right]


class VideoPairDataset(Dataset):
    """Compressed-video backend with online delta and worker-local chunk cache."""

    def __init__(
        self,
        videos: Sequence[VideoRecord],
        *,
        source_root: str | Path,
        split_name: str,
        delta: int | Sequence[int],
        image_size: tuple[int, int],
        augment: PairedAugment | None = None,
        delta_probabilities: Sequence[float] | None = None,
        online_delta: bool = False,
        chunk_frames: int = 128,
        cache_chunks: int = 2,
        ffmpeg_bin: str = "ffmpeg",
        ffmpeg_threads: int = 2,
    ) -> None:
        if not videos:
            raise ValueError(f"{split_name} video dataset has no videos")
        self.root = str(Path(source_root).resolve())
        self.split_name = str(split_name)
        self.deltas = _normalize_deltas(delta)
        self.image_size = tuple(int(x) for x in image_size)
        self.augment = augment
        self.online_delta = bool(online_delta and len(self.deltas) > 1)
        self.delta_probabilities = _normalize_delta_probabilities(self.deltas, delta_probabilities)
        self.chunk_frames = int(chunk_frames)
        self.cache_chunks = int(cache_chunks)
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.ffmpeg_threads = int(ffmpeg_threads)
        if self.chunk_frames <= max(self.deltas):
            raise ValueError("video.chunk_frames must be larger than max delta")
        if self.cache_chunks <= 0 or self.ffmpeg_threads <= 0:
            raise ValueError("video.cache_chunks and ffmpeg_threads must be positive")

        max_delta = max(self.deltas)
        filtered: list[VideoRecord] = []
        counts: list[int] = []
        for video in videos:
            count = max(0, video.frame_count - max_delta)
            if count > 0:
                if (video.height, video.width) != self.image_size:
                    raise ValueError(
                        f"Video {video.path} has size {(video.height, video.width)}, "
                        f"expected {self.image_size}"
                    )
                filtered.append(video)
                counts.append(count)
        if not filtered:
            raise ValueError(f"{split_name} has no legal video starts for delta={self.deltas}")

        self.videos = tuple(filtered)
        self._counts = tuple(counts)
        self._ends: list[int] = []
        total = 0
        for count in counts:
            total += count
            self._ends.append(total)
        self._length = total
        self._reader: _VideoChunkCache | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[int, VideoRecord, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        video_index = bisect.bisect_right(self._ends, index)
        previous = 0 if video_index == 0 else self._ends[video_index - 1]
        return video_index, self.videos[video_index], index - previous

    def _choose_delta(self) -> int:
        if len(self.deltas) == 1:
            return self.deltas[0]
        sampled = int(torch.multinomial(self.delta_probabilities, 1, replacement=True).item())
        return self.deltas[sampled]

    def _get_reader(self) -> _VideoChunkCache:
        if self._reader is None:
            self._reader = _VideoChunkCache(
                image_size=self.image_size,
                chunk_frames=self.chunk_frames,
                cache_chunks=self.cache_chunks,
                ffmpeg_bin=self.ffmpeg_bin,
                ffmpeg_threads=self.ffmpeg_threads,
            )
        return self._reader

    def __getitem__(self, index: int) -> dict:
        _, video, left = self._locate(index)
        delta = self._choose_delta()
        image0, image1 = self._get_reader().get(video, left, left + delta)
        images = torch.stack((image0, image1), dim=0)
        if self.augment is not None:
            images = self.augment(images)
        return {
            "images": images,
            "label": video.label,
            "game": video.game,
            "video_id": video.video_id,
            "delta": delta,
        }

    @property
    def video_keys(self) -> set[tuple[str, int, str]]:
        return {video.key for video in self.videos}

    def position_count_for_video(self, index: int) -> int:
        return self._counts[index]

    def summary(self) -> dict:
        class_positions = {0: 0, 1: 0}
        games: set[str] = set()
        for index, video in enumerate(self.videos):
            class_positions[video.label] += self.position_count_for_video(index)
            games.add(video.game)
        return {
            "backend": "video",
            "split": self.split_name,
            "root": self.root,
            "sampling_positions": len(self),
            "pairs": len(self),
            "videos": len(self.videos),
            "games": len(games),
            "class_positions": class_positions,
            "deltas": list(self.deltas),
            "delta_mode": "online" if self.online_delta else "fixed",
            "delta_probabilities": {
                str(delta): float(prob)
                for delta, prob in zip(self.deltas, self.delta_probabilities.tolist(), strict=True)
            },
            "image_size": list(self.image_size),
            "chunk_frames": self.chunk_frames,
            "cache_chunks": self.cache_chunks,
            "augmentation": self.augment is not None,
        }


def build_video_datasets(config: dict):
    data_cfg = config["data"]
    video_cfg = data_cfg.get("video", {}) or {}
    image_size = tuple(int(x) for x in data_cfg["image_size"])
    train_deltas = _train_deltas(data_cfg)
    eval_delta = int(data_cfg["eval_delta"])
    train_root, test_root = _backend_roots(data_cfg, "video")
    if Path(train_root).resolve() == Path(test_root).resolve():
        raise ValueError("video train_root and test_root must be different directories")

    common_scan = {
        "extensions": video_cfg.get("extensions", [".mp4"]),
        "ffprobe_bin": str(video_cfg.get("ffprobe_bin", "ffprobe")),
    }
    source_videos = scan_video_root(train_root, **common_scan)
    split_candidates = [
        video for video in source_videos if video.pair_count_for_delta(eval_delta) > 0
    ]
    train_videos, val_videos = split_train_val_videos(
        split_candidates,
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(config["experiment"]["seed"]),
    )
    test_videos = scan_video_root(test_root, **common_scan)
    test_videos = [video for video in test_videos if video.pair_count_for_delta(eval_delta) > 0]

    train_augment = build_train_augment(config.get("augment"))
    common_dataset = {
        "image_size": image_size,
        "chunk_frames": int(video_cfg.get("chunk_frames", 128)),
        "cache_chunks": int(video_cfg.get("cache_chunks", 2)),
        "ffmpeg_bin": str(video_cfg.get("ffmpeg_bin", "ffmpeg")),
        "ffmpeg_threads": int(video_cfg.get("ffmpeg_threads", 2)),
    }
    return (
        VideoPairDataset(
            train_videos,
            source_root=train_root,
            split_name="train",
            delta=train_deltas,
            delta_probabilities=data_cfg.get("train_delta_probabilities"),
            online_delta=True,
            augment=train_augment,
            **common_dataset,
        ),
        VideoPairDataset(
            val_videos,
            source_root=train_root,
            split_name="val",
            delta=eval_delta,
            **common_dataset,
        ),
        VideoPairDataset(
            test_videos,
            source_root=test_root,
            split_name="test",
            delta=eval_delta,
            **common_dataset,
        ),
    )
