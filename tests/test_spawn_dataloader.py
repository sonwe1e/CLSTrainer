from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError:
    np = torch = Image = None

from game_cls.data.collate import pair_collate


class SpawnDataset:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        import torch

        return {
            "images": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
            "label": index % 2,
            "meta": {"index": index},
        }


@unittest.skipIf(torch is None, "torch is not installed")
class SpawnDataLoaderTests(unittest.TestCase):
    def test_spawn_loader_returns_batch(self) -> None:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            SpawnDataset(),
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
            timeout=20,
            collate_fn=pair_collate,
        )

        batches = list(loader)

        self.assertEqual(len(batches), 4)
        self.assertEqual(
            tuple(batches[0]["images"].shape), (2, 2, 3, 8, 8)
        )

    @unittest.skipIf(
        np is None or Image is None,
        "PNG spawn test dependencies are not installed",
    )
    def test_real_png_pair_dataset_with_augmentation_uses_spawn(self) -> None:
        from torch.utils.data import DataLoader

        from game_cls.data.augment import ConsistentPairAugment
        from game_cls.data.lazy_pair_dataset import LazyTrainingPairDataset
        from game_cls.data.video_index import VideoEntry
        from game_cls.data.video_sampler import (
            VideoBalancedPairBatchSampler,
        )

        with tempfile.TemporaryDirectory() as directory:
            video_directory = Path(directory) / "gameA" / "0"
            video_directory.mkdir(parents=True)
            for frame_id in range(3):
                pixels = np.full(
                    (8, 8, 3), frame_id * 40, dtype=np.uint8
                )
                Image.fromarray(pixels).save(
                    video_directory / f"01{frame_id:05d}.png"
                )
            video = VideoEntry(
                game="gameA",
                label=0,
                video_id="01",
                frame_ids=np.asarray([0, 1, 2], dtype=np.int32),
                valid_start_positions={
                    1: np.asarray([0, 1], dtype=np.int32)
                },
                video_directory=str(video_directory),
            )
            transform = ConsistentPairAugment(
                {
                    "color_jitter": {
                        "enabled": True,
                        "probability": 1.0,
                        "brightness": 0.1,
                        "contrast": 0.1,
                    }
                }
            )
            dataset = LazyTrainingPairDataset(
                [video], transform=transform
            )
            sampler = VideoBalancedPairBatchSampler(
                [video],
                local_batch_size=2,
                steps_per_epoch=1,
                delta_probability={1: 1.0},
                seed=7,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=1,
                multiprocessing_context="spawn",
                timeout=20,
                collate_fn=pair_collate,
            )

            batches = list(loader)

            self.assertEqual(len(batches), 1)
            self.assertEqual(
                tuple(batches[0]["images"].shape), (2, 2, 3, 8, 8)
            )
            self.assertEqual(batches[0]["images"].dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
