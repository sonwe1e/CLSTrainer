from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the current interpreter")
class TrainingSmokeTests(unittest.TestCase):
    def test_cls_learns_while_backbone_is_bitwise_frozen(self) -> None:
        from game_cls.model.builder import build_demo_model
        from game_cls.model.freeze_policy import (
            assert_frozen_parameters_unchanged,
            configure_trainable_parameters,
            snapshot_frozen_parameters,
        )

        torch.manual_seed(3)
        model = build_demo_model({})
        configure_trainable_parameters(model)
        frozen = snapshot_frozen_parameters(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.1,
        )
        images = torch.zeros(4, 2, 3, 448, 208)
        images[2:, :, 0] = 1.0
        targets = torch.tensor([0, 0, 1, 1])
        with torch.no_grad():
            initial = torch.nn.functional.cross_entropy(
                model(images[:, 0], images[:, 1]), targets
            ).item()
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(images[:, 0], images[:, 1]), targets
            )
            loss.backward()
            optimizer.step()
        final = torch.nn.functional.cross_entropy(
            model(images[:, 0], images[:, 1]), targets
        ).item()
        self.assertEqual(tuple(model(images[:, 0], images[:, 1]).shape), (4, 2))
        self.assertLess(final, initial * 0.1)
        assert_frozen_parameters_unchanged(frozen, model)


@unittest.skipIf(torch is None, "torch is not installed in the current interpreter")
class PairAugmentTests(unittest.TestCase):
    def test_identical_frames_receive_identical_random_transform(self) -> None:
        from game_cls.data.augment import ConsistentPairAugment

        transform = ConsistentPairAugment(
            {
                "random_affine": {
                    "enabled": True,
                    "probability": 1.0,
                    "degrees": 5.0,
                    "translate": [0.05, 0.05],
                    "scale": [0.95, 1.05],
                    "shear": [-2.0, 2.0],
                },
                "color_jitter": {
                    "enabled": True,
                    "probability": 1.0,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.02,
                },
                "random_erasing": {
                    "enabled": True,
                    "probability": 1.0,
                    "scale": [0.01, 0.05],
                    "ratio": [0.5, 2.0],
                    "value": "random",
                },
            }
        )
        frame = torch.randint(0, 256, (3, 64, 32), dtype=torch.uint8)
        pair = torch.stack([frame, frame])
        for _ in range(10):
            output = transform(pair)
            self.assertEqual(output.dtype, torch.uint8)
            self.assertTrue(torch.equal(output[0], output[1]))

    def test_uint8_random_erasing_uses_full_rgb_range(self) -> None:
        from game_cls.data.augment import ConsistentPairAugment

        transform = ConsistentPairAugment(
            {
                "random_erasing": {
                    "enabled": True,
                    "probability": 1.0,
                    "scale": [0.25, 0.25],
                    "ratio": [1.0, 1.0],
                    "value": "random",
                }
            }
        )
        frame = torch.zeros(3, 32, 32, dtype=torch.uint8)
        output = transform(torch.stack([frame, frame]))
        self.assertTrue(torch.equal(output[0], output[1]))
        self.assertGreater(int(output.max()), 32)
        self.assertGreater(torch.unique(output).numel(), 16)


if __name__ == "__main__":
    unittest.main()
