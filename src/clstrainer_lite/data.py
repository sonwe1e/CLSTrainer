from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .augments import PairedAugment, build_train_augment


FRAME_RE = re.compile(r"^(?P<video>\d{2})(?P<frame>\d{5})\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageVideo:
    """One source video represented by its extracted PNG frames."""

    game: str
    label: int
    video_id: str
    frame_ids: tuple[int, ...]
    paths: tuple[str, ...]

    @property
    def key(self) -> tuple[str, int, str]:
        return self.game, self.label, self.video_id

    @property
    def frame_count(self) -> int:
        return len(self.paths)

    def pair_count_for_delta(self, delta: int) -> int:
        delta = int(delta)
        ids = set(self.frame_ids)
        return sum((frame_id + delta) in ids for frame_id in self.frame_ids)


def _normalize_deltas(delta: int | Sequence[int]) -> tuple[int, ...]:
    values = (int(delta),) if isinstance(delta, int) else tuple(int(x) for x in delta)
    values = tuple(sorted(set(values)))
    if not values or any(value <= 0 for value in values):
        raise ValueError("delta values must be positive integers")
    return values


def _normalize_delta_probabilities(
    deltas: Sequence[int], probabilities: Sequence[float] | None
) -> torch.Tensor:
    if probabilities is None:
        return torch.full((len(deltas),), 1.0 / len(deltas), dtype=torch.float64)
    values = torch.tensor([float(x) for x in probabilities], dtype=torch.float64)
    if values.numel() != len(deltas):
        raise ValueError("train_delta_probabilities length must equal number of train deltas")
    if bool((values < 0).any()) or float(values.sum()) <= 0:
        raise ValueError("train_delta_probabilities must be non-negative and sum to > 0")
    return values / values.sum()


def _load_rgb(path: str, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    chw = np.transpose(array, (2, 0, 1)).copy()
    return torch.from_numpy(chw)


def scan_image_root(
    root: str | Path,
    *,
    strict_filenames: bool = True,
) -> list[ImageVideo]:
    """Scan ``<root>/<game>/<0|1>/<VV><FFFFF>.png`` without expanding pairs.

    Only frame metadata is indexed.  Delta is selected later by
    :class:`ImagePairDataset`, so a train delta range no longer multiplies the
    dataset index at startup.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    videos: list[ImageVideo] = []
    for game_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for label in (0, 1):
            label_dir = game_dir / str(label)
            if not label_dir.is_dir():
                continue
            by_video: dict[str, list[tuple[int, str]]] = {}
            for path in sorted(label_dir.glob("*.png")):
                match = FRAME_RE.fullmatch(path.name)
                if match is None:
                    if strict_filenames:
                        raise ValueError(
                            f"Invalid frame filename {path}; expected VVFFFFF.png "
                            "(2 digit video id + 5 digit frame id)."
                        )
                    continue
                video_id = match.group("video")
                frame_id = int(match.group("frame"))
                by_video.setdefault(video_id, []).append((frame_id, str(path)))

            for video_id, rows in sorted(by_video.items()):
                rows.sort(key=lambda item: item[0])
                frame_ids = tuple(frame_id for frame_id, _ in rows)
                if len(frame_ids) != len(set(frame_ids)):
                    raise ValueError(f"Duplicate frame ids in {game_dir.name}/{label}/{video_id}")
                videos.append(
                    ImageVideo(
                        game=game_dir.name,
                        label=label,
                        video_id=video_id,
                        frame_ids=frame_ids,
                        paths=tuple(path for _, path in rows),
                    )
                )
    if not videos:
        raise ValueError(f"No PNG videos found under {root}")
    return videos


def scan_pair_root(
    root: str | Path,
    *,
    delta: int | Sequence[int],
    strict_filenames: bool = True,
) -> list[ImageVideo]:
    """Compatibility wrapper for 0.3 callers.

    It no longer creates PairPosition objects.  It only filters videos that can
    produce at least one requested delta.
    """

    deltas = _normalize_deltas(delta)
    videos = scan_image_root(root, strict_filenames=strict_filenames)
    filtered = [
        video
        for video in videos
        if any(video.pair_count_for_delta(current_delta) > 0 for current_delta in deltas)
    ]
    if not filtered:
        raise ValueError(f"No legal frame pairs found under {root} for delta={deltas}")
    return filtered


def _split_score(video, seed: int) -> str:
    text = f"{seed}|{video.game}|{video.label}|{video.video_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_train_val_videos(videos, *, val_ratio: float, seed: int):
    """Deterministically split by complete video, approximately game/class stratified."""

    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    if len(videos) < 2:
        raise ValueError("Need at least two videos to derive validation from train_root")

    groups: dict[tuple[str, int], list] = {}
    for video in videos:
        groups.setdefault((video.game, video.label), []).append(video)

    total = len(videos)
    requested = max(1, min(total - 1, int(round(total * float(val_ratio)))))
    capacities = {key: max(0, len(rows) - 1) for key, rows in groups.items()}
    capacity = sum(capacities.values())
    if capacity == 0:
        raise ValueError(
            "Cannot derive a leak-free validation split: every (game, label) stratum "
            "has only one video."
        )

    labels = sorted({video.label for video in videos})
    label_capacity = {
        label: sum(cap for (_, group_label), cap in capacities.items() if group_label == label)
        for label in labels
    }
    missing_capacity = [label for label in labels if label_capacity[label] == 0]
    if missing_capacity:
        raise ValueError(
            "Cannot put every observed class into both train and validation while keeping "
            "whole videos intact. Add another video for label(s): "
            + ", ".join(str(label) for label in missing_capacity)
        )

    target = min(max(requested, len(labels)), capacity)
    ideal = {key: target * len(rows) / total for key, rows in groups.items()}
    quota = {key: 0 for key in groups}

    for label in labels:
        candidates = [key for key in groups if key[1] == label and quota[key] < capacities[key]]
        candidates.sort(key=lambda key: (ideal[key], len(groups[key]), str(key)), reverse=True)
        quota[candidates[0]] += 1

    remaining = target - sum(quota.values())
    while remaining > 0:
        candidates = [key for key in groups if quota[key] < capacities[key]]
        if not candidates:
            break
        candidates.sort(key=lambda key: (ideal[key] - quota[key], str(key)), reverse=True)
        quota[candidates[0]] += 1
        remaining -= 1

    train_videos: list = []
    val_videos: list = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda video: _split_score(video, seed))
        count = quota[key]
        val_videos.extend(ranked[:count])
        train_videos.extend(ranked[count:])

    if not train_videos or not val_videos:
        raise RuntimeError("Internal error: train/validation split became empty")
    return train_videos, val_videos


class ImagePairDataset(Dataset):
    """PNG-backed dual-frame dataset with online delta selection.

    The index contains one compact *start position* per sample, not one object
    for every ``(start, delta)`` combination.  For training, all configured
    deltas are valid at every indexed start and one is sampled in ``__getitem__``.
    Validation/test normally use one fixed delta.
    """

    def __init__(
        self,
        videos: Sequence[ImageVideo],
        *,
        source_root: str | Path,
        split_name: str,
        delta: int | Sequence[int],
        image_size: tuple[int, int],
        augment: PairedAugment | None = None,
        delta_probabilities: Sequence[float] | None = None,
        online_delta: bool = False,
    ) -> None:
        if not videos:
            raise ValueError(f"{split_name} dataset has no videos")
        self.root = str(Path(source_root).resolve())
        self.split_name = str(split_name)
        self.deltas = _normalize_deltas(delta)
        self.image_size = tuple(int(x) for x in image_size)
        self.augment = augment
        self.online_delta = bool(online_delta and len(self.deltas) > 1)
        self.delta_probabilities = _normalize_delta_probabilities(
            self.deltas, delta_probabilities
        )

        filtered_videos: list[ImageVideo] = []
        starts: list[np.ndarray] = []
        for video in videos:
            ids = set(video.frame_ids)
            # Requiring every configured train delta to exist at an indexed
            # start keeps the requested online delta distribution exact.
            valid = [
                index
                for index, frame_id in enumerate(video.frame_ids)
                if all((frame_id + current_delta) in ids for current_delta in self.deltas)
            ]
            if valid:
                filtered_videos.append(video)
                starts.append(np.asarray(valid, dtype=np.int32))
        if not filtered_videos:
            raise ValueError(f"{split_name} dataset has no legal starts for delta={self.deltas}")

        self.videos = tuple(filtered_videos)
        self._starts = tuple(starts)
        self._ends: list[int] = []
        total = 0
        for rows in self._starts:
            total += int(rows.size)
            self._ends.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[int, ImageVideo, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        video_index = bisect.bisect_right(self._ends, index)
        previous_end = 0 if video_index == 0 else self._ends[video_index - 1]
        local_index = index - previous_end
        left = int(self._starts[video_index][local_index])
        return video_index, self.videos[video_index], left

    def _choose_delta(self) -> int:
        if len(self.deltas) == 1:
            return self.deltas[0]
        sampled = int(torch.multinomial(self.delta_probabilities, 1, replacement=True).item())
        return self.deltas[sampled]

    def __getitem__(self, index: int) -> dict:
        _, video, left = self._locate(index)
        delta = self._choose_delta()
        target_frame_id = video.frame_ids[left] + delta
        right = bisect.bisect_left(video.frame_ids, target_frame_id)
        if right >= len(video.frame_ids) or video.frame_ids[right] != target_frame_id:
            raise RuntimeError("Internal error: indexed image start cannot satisfy sampled delta")

        image0 = _load_rgb(video.paths[left], self.image_size)
        image1 = _load_rgb(video.paths[right], self.image_size)
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
        return int(self._starts[index].size)

    def summary(self) -> dict:
        class_positions = {0: 0, 1: 0}
        games: set[str] = set()
        for index, video in enumerate(self.videos):
            class_positions[video.label] += self.position_count_for_video(index)
            games.add(video.game)
        return {
            "backend": "image",
            "split": self.split_name,
            "root": self.root,
            "sampling_positions": len(self),
            "pairs": len(self),  # compatibility: one sampled pair per position per epoch
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
            "augmentation": self.augment is not None,
        }


# Temporary compatibility alias.  New code should use ImagePairDataset.
PairDataset = ImagePairDataset


def _train_deltas(data_cfg: dict) -> tuple[int, ...]:
    start, end = (int(x) for x in data_cfg["train_delta_range"])
    return tuple(range(start, end + 1))


def _backend_roots(data_cfg: dict, backend: str) -> tuple[str, str]:
    backend_cfg = data_cfg.get(backend, {}) or {}
    # Backward compatibility for 0.3-style direct dictionaries.
    train_root = backend_cfg.get("train_root") or data_cfg.get("train_root")
    test_root = backend_cfg.get("test_root") or data_cfg.get("test_root")
    if not train_root or not test_root:
        raise ValueError(f"data.{backend}.train_root and test_root are required for backend={backend}")
    return str(train_root), str(test_root)


def build_image_datasets(config: dict) -> tuple[ImagePairDataset, ImagePairDataset, ImagePairDataset]:
    data_cfg = config["data"]
    image_cfg = data_cfg.get("image", {}) or {}
    image_size = tuple(int(x) for x in data_cfg["image_size"])
    train_deltas = _train_deltas(data_cfg)
    eval_delta = int(data_cfg["eval_delta"])
    strict = bool(image_cfg.get("strict_filenames", data_cfg.get("strict_filenames", True)))
    train_root, test_root = _backend_roots(data_cfg, "image")
    if Path(train_root).resolve() == Path(test_root).resolve():
        raise ValueError("image train_root and test_root must be different directories")

    source_videos = scan_image_root(train_root, strict_filenames=strict)
    split_candidates = [
        video for video in source_videos if video.pair_count_for_delta(eval_delta) > 0
    ]
    train_videos, val_videos = split_train_val_videos(
        split_candidates,
        val_ratio=float(data_cfg["val_ratio"]),
        seed=int(config["experiment"]["seed"]),
    )
    test_videos = scan_image_root(test_root, strict_filenames=strict)
    test_videos = [video for video in test_videos if video.pair_count_for_delta(eval_delta) > 0]
    train_augment = build_train_augment(config.get("augment"))

    probabilities = data_cfg.get("train_delta_probabilities")
    return (
        ImagePairDataset(
            train_videos,
            source_root=train_root,
            split_name="train",
            delta=train_deltas,
            delta_probabilities=probabilities,
            online_delta=True,
            image_size=image_size,
            augment=train_augment,
        ),
        ImagePairDataset(
            val_videos,
            source_root=train_root,
            split_name="val",
            delta=eval_delta,
            image_size=image_size,
        ),
        ImagePairDataset(
            test_videos,
            source_root=test_root,
            split_name="test",
            delta=eval_delta,
            image_size=image_size,
        ),
    )


def build_datasets(config: dict):
    backend = str(config["data"].get("backend", "image")).lower()
    if backend == "image":
        return build_image_datasets(config)
    if backend == "video":
        from .video_data import build_video_datasets

        return build_video_datasets(config)
    raise ValueError(f"Unsupported data.backend={backend!r}; expected image or video")


class DistributedEvalSampler(Sampler[int]):
    """Shard validation/test samples across ranks without padding/duplication."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.length = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world_size + 1
