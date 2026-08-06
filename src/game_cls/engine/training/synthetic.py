from __future__ import annotations

from game_cls.data.image_spec import ImageSpec


class SyntheticPairDataset:
    def __init__(self, length: int, image_spec: ImageSpec, seed: int) -> None:
        self.length = length
        self.image_spec = image_spec
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        import torch

        generator = torch.Generator().manual_seed(self.seed + index)
        label = index % 2
        images = torch.randint(
            0,
            96,
            (2, *self.image_spec.chw),
            dtype=torch.uint8,
            generator=generator,
        )
        if label:
            images[:, 0, : self.image_spec.height // 2] += 128
        return {
            "images": images,
            "label": label,
            "meta": {
                "game": f"synthetic_{index % 2}",
                "video_id": f"{index % 4:02d}",
                "frame0_id": index,
                "frame1_id": index + 2,
                "delta": 2,
                "image0_path": "",
                "image1_path": "",
            },
        }
