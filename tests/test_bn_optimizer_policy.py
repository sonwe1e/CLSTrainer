from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class BatchNormAndOptimizerTests(unittest.TestCase):
    def test_backbone_and_cls_batchnorm_modes_are_independent(self) -> None:
        from game_cls.model.freeze_policy import set_frozen_backbone_train_mode

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3)
                )
                self.cls = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3)
                )

        model = Model()
        set_frozen_backbone_train_mode(
            model,
            freeze_backbone_batchnorm_stats=True,
            freeze_cls_batchnorm_stats=False,
        )
        self.assertFalse(model.backbone[1].training)
        self.assertTrue(model.cls[1].training)

    def test_bias_and_one_dimensional_parameters_have_zero_decay(self) -> None:
        from game_cls.engine.trainer import build_optimizer_parameter_groups
        from game_cls.model.freeze_policy import configure_trainable_parameters

        model = torch.nn.Module()
        model.cls = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        configure_trainable_parameters(model)
        groups = build_optimizer_parameter_groups(model, 0.01)
        by_decay = {group["weight_decay"]: group["params"] for group in groups}
        self.assertEqual(len(by_decay[0.01]), 1)
        self.assertEqual(len(by_decay[0.0]), 3)

    def test_ddp_rejects_unsynchronized_cls_batchnorm_statistics(self) -> None:
        from game_cls.config import load_config
        from game_cls.engine.trainer import validate_training_config

        config = load_config("configs/recipes/example_debug.yaml")
        config.setdefault("distributed", {})["enabled"] = True
        config["model"]["freeze_cls_batchnorm_stats"] = False
        with self.assertRaisesRegex(RuntimeError, "SyncBatchNorm"):
            validate_training_config(config)


if __name__ == "__main__":
    unittest.main()
