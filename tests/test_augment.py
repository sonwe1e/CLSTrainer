"""Stage-4 pair augmentation: two-frame-shared transforms and defaults.

Every new transform must draw ONE shared parameter and apply it to both
frames of a stacked [2,C,H,W] pair, so feeding a pair of identical frames
must yield identical outputs. All new transforms also default to disabled,
which means the augmenter must be an exact identity (no raise, no change).
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

# Conservative per-block configs for each new transform, keyed by the
# augmentation.dict block name (image spec: height 32, width 48).
_NEW_TRANSFORM_BLOCKS: dict[str, dict] = {
    "random_perspective": {
        "enabled": True,
        "probability": 1.0,
        "distortion_scale": 0.1,
    },
    "random_resized_crop": {
        "enabled": True,
        "probability": 1.0,
        "scale": [0.9, 1.0],
        "ratio": [0.9, 1.1],
    },
    "gamma": {
        "enabled": True,
        "probability": 1.0,
        "gamma_range": [0.8, 1.2],
    },
    "exposure": {
        "enabled": True,
        "probability": 1.0,
        "factor_range": [0.85, 1.15],
    },
    "blur": {
        "enabled": True,
        "probability": 1.0,
        "kernel_size": 3,
        "sigma_range": [0.1, 1.0],
    },
    "noise": {
        "enabled": True,
        "probability": 1.0,
        "noise_std": 0.02,
    },
    "jpeg_compression": {
        "enabled": True,
        "probability": 1.0,
        "quality_range": [60, 95],
    },
}

# A full augmentation block with every new transform explicitly disabled; the
# master switch and the inactive transforms are irrelevant to these tests.
_ALL_DISABLED: dict[str, dict] = {
    name: {"enabled": False, **{k: v for k, v in block.items() if k != "enabled"}}
    for name, block in _NEW_TRANSFORM_BLOCKS.items()
}


@unittest.skipIf(torch is None, "torch is not installed in the current interpreter")
class ConsistentPairAugmentStage4Tests(unittest.TestCase):
    def _assert_shared(self, config: dict) -> None:
        from game_cls.data.augment import ConsistentPairAugment

        transform = ConsistentPairAugment({"enabled": True, **config})
        for _ in range(5):
            torch.manual_seed(0)
            frame = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
            output = transform(torch.stack([frame, frame]))
            self.assertEqual(tuple(output.shape), (2, 3, 32, 48))
            self.assertEqual(output.dtype, torch.uint8)
            self.assertTrue(torch.equal(output[0], output[1]))

    def test_each_new_transform_shares_one_draw(self) -> None:
        for name, block in _NEW_TRANSFORM_BLOCKS.items():
            with self.subTest(transform=name):
                self._assert_shared({name: block})

    def test_random_resized_crop_with_explicit_size_is_shared(self) -> None:
        from game_cls.data.augment import ConsistentPairAugment

        block = {
            **_NEW_TRANSFORM_BLOCKS["random_resized_crop"],
            "size": [32, 48],
        }
        transform = ConsistentPairAugment(
            {"enabled": True, "random_resized_crop": block}
        )
        torch.manual_seed(0)
        frame = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
        output = transform(torch.stack([frame, frame]))
        self.assertEqual(tuple(output.shape), (2, 3, 32, 48))
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_all_transforms_disabled_is_identity(self) -> None:
        from game_cls.data.augment import ConsistentPairAugment

        transform = ConsistentPairAugment({"enabled": True, **_ALL_DISABLED})
        frame = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
        pair = torch.stack([frame, frame])
        output = transform(pair)
        self.assertTrue(torch.equal(output, pair))
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_empty_augmentation_dict_does_not_raise(self) -> None:
        from game_cls.data.augment import ConsistentPairAugment

        transform = ConsistentPairAugment({})
        frame = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
        output = transform(torch.stack([frame, frame]))
        self.assertTrue(torch.equal(output[0], output[1]))


if __name__ == "__main__":
    unittest.main()
