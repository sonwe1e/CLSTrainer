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


@unittest.skipIf(torch is None, "training dependencies are not installed")
class MultiWorkerResumeTests(unittest.TestCase):
    def test_seeded_augmentation_matches_after_worker_restart(self) -> None:
        from torch.utils.data import DataLoader

        from game_cls.data.augment import ConsistentPairAugment
        from game_cls.data.collate import pair_collate
        from game_cls.data.lazy_pair_dataset import LazyTrainingPairDataset
        from game_cls.data.records import FrameRecord
        from game_cls.data.video_index import build_video_entries
        from game_cls.data.video_sampler import VideoBalancedPairBatchSampler

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for label in (0, 1):
                for frame_id in range(1, 8):
                    array = np.full((32, 16, 3), label * 80 + frame_id, dtype=np.uint8)
                    path = root / str(label) / f"01{frame_id:05d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(path)
                    frames.append(
                        FrameRecord(
                            sample_id=f"{label}:{frame_id}",
                            split="train",
                            game="A",
                            label=label,
                            video_id="01",
                            frame_id=frame_id,
                            path=str(path),
                            width=16,
                            height=32,
                            channels=3,
                            file_size=path.stat().st_size,
                        )
                    )
            videos = build_video_entries(frames)
            transform = ConsistentPairAugment(
                {
                    "random_affine": {
                        "enabled": True,
                        "probability": 1.0,
                        "degrees": 3.0,
                        "translate": [0.03, 0.03],
                        "scale": [0.98, 1.02],
                        "shear": [-1.0, 1.0],
                    },
                    "random_erasing": {
                        "enabled": True,
                        "probability": 1.0,
                        "scale": [0.05, 0.05],
                        "ratio": [1.0, 1.0],
                        "value": "random",
                    },
                }
            )

            def collect(start_step: int):
                dataset = LazyTrainingPairDataset(videos, transform=transform)
                sampler = VideoBalancedPairBatchSampler(
                    videos,
                    local_batch_size=2,
                    steps_per_epoch=4,
                    seed=123,
                    delta_probability={1: 0.0, 2: 1.0, 3: 0.0},
                )
                sampler.set_epoch(0, start_step=start_step)
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    num_workers=2,
                    collate_fn=pair_collate,
                )
                return [batch["images"].clone() for batch in loader]

            uninterrupted = collect(0)
            resumed = collect(2)
            self.assertEqual(len(resumed), 2)
            for expected, actual in zip(uninterrupted[2:], resumed, strict=False):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
