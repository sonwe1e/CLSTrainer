from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .video_index import VideoEntry


@dataclass(frozen=True)
class PairRequest:
    video_index: int
    delta: int
    start_position: int
    augmentation_seed: int = 0


def _decode_image_png(path: str):
    try:
        from PIL import Image
        from torchvision.transforms.v2 import functional as F
    except ImportError as exc:
        raise RuntimeError(
            "Decoding pairs requires torch, torchvision and Pillow"
        ) from exc
    with Image.open(path) as image:
        return F.to_image(image.convert("RGB"))


def _decode_pair(entry: VideoEntry, request: PairRequest, decoder):
    import torch

    frame0_id, frame1_id, path0, path1 = entry.pair_paths(
        request.delta, request.start_position
    )
    tensor0 = decoder(path0)
    tensor1 = decoder(path1)
    def display(reference) -> str:
        return (
            reference
            if isinstance(reference, str)
            else f"packed://frame/{int(reference)}"
        )
    return (
        torch.stack((tensor0, tensor1), dim=0),
        {
            "game": entry.game,
            "label": entry.label,
            "video_id": entry.video_id,
            "frame0_id": frame0_id,
            "frame1_id": frame1_id,
            "delta": request.delta,
            "image0_path": display(path0),
            "image1_path": display(path1),
        },
    )


class LazyTrainingPairDataset:
    """Dataset addressed by compact PairRequest objects emitted by the sampler."""

    def __init__(
        self,
        videos: Sequence[VideoEntry],
        transform: Callable[[Any], Any] | None = None,
        decoder: Callable[[Any], Any] | None = None,
    ) -> None:
        self.videos = videos
        self.transform = transform
        self.decoder = decoder or _decode_image_png

    def __len__(self) -> int:
        return sum(
            len(starts)
            for video in self.videos
            for starts in video.valid_start_positions.values()
        )

    def __getitem__(self, request: PairRequest) -> dict[str, Any]:
        import torch

        images, meta = _decode_pair(
            self.videos[request.video_index], request, self.decoder
        )
        if self.transform is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(request.augmentation_seed)
                images = self.transform(images)
        return {"images": images, "label": meta["label"], "meta": meta}


class EvalPairDataset:
    """Compact, rank-local evaluation pair index backed by NumPy arrays."""

    def __init__(
        self,
        videos: Sequence[VideoEntry],
        video_indices: np.ndarray,
        deltas: np.ndarray,
        start_positions: np.ndarray,
        decoder: Callable[[Any], Any] | None = None,
    ) -> None:
        self.videos = videos
        self.video_indices = video_indices.astype(np.int32, copy=False)
        self.deltas = deltas.astype(np.int8, copy=False)
        self.start_positions = start_positions.astype(np.int32, copy=False)
        self.decoder = decoder or _decode_image_png
        game_keys = sorted({video.game for video in videos})
        game_label_keys = sorted(
            {(video.game, video.label) for video in videos}
        )
        self.group_catalogs = {
            "game": game_keys,
            "game_label": game_label_keys,
            "video": [
                (video.game, video.label, video.video_id)
                for video in videos
            ],
        }
        game_to_id = {key: index for index, key in enumerate(game_keys)}
        game_label_to_id = {
            key: index for index, key in enumerate(game_label_keys)
        }
        self._game_id_by_video = np.asarray(
            [game_to_id[video.game] for video in videos], dtype=np.int32
        )
        self._game_label_id_by_video = np.asarray(
            [
                game_label_to_id[(video.game, video.label)]
                for video in videos
            ],
            dtype=np.int32,
        )

    def __len__(self) -> int:
        return len(self.video_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        request = PairRequest(
            video_index=int(self.video_indices[index]),
            delta=int(self.deltas[index]),
            start_position=int(self.start_positions[index]),
        )
        images, meta = _decode_pair(
            self.videos[request.video_index], request, self.decoder
        )
        return {
            "images": images,
            "label": meta["label"],
            "game_id": int(
                self._game_id_by_video[request.video_index]
            ),
            "game_label_id": int(
                self._game_label_id_by_video[request.video_index]
            ),
            "video_group_id": request.video_index,
            "meta": meta,
        }

    @property
    def index_nbytes(self) -> int:
        return (
            self.video_indices.nbytes
            + self.deltas.nbytes
            + self.start_positions.nbytes
        )


def build_eval_dataset(
    videos: Sequence[VideoEntry],
    delta: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    max_pairs_per_video: int | None = None,
    decoder: Callable[[Any], Any] | None = None,
) -> EvalPairDataset:
    video_indices: list[int] = []
    deltas: list[int] = []
    start_positions: list[int] = []
    global_pair_index = 0
    for video_index, video in enumerate(videos):
        starts = video.valid_start_positions.get(delta, np.empty(0, dtype=np.int32))
        if max_pairs_per_video is not None and len(starts) > max_pairs_per_video:
            selected = np.linspace(
                0, len(starts) - 1, num=max_pairs_per_video, dtype=np.int64
            )
            starts = starts[selected]
        for start in starts:
            if global_pair_index % world_size == rank:
                video_indices.append(video_index)
                deltas.append(delta)
                start_positions.append(int(start))
            global_pair_index += 1
    return EvalPairDataset(
        videos,
        np.asarray(video_indices, dtype=np.int32),
        np.asarray(deltas, dtype=np.int8),
        np.asarray(start_positions, dtype=np.int32),
        decoder=decoder,
    )
