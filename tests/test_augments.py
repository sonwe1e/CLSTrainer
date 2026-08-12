import torch

from clstrainer_lite.augments import PairedAugment, PairedAugmentConfig
from clstrainer_lite.losses import FocalLoss


def test_paired_augment_keeps_shared_pair_geometry_and_color_identical():
    torch.manual_seed(123)
    frame = torch.arange(3 * 20 * 30, dtype=torch.int64).remainder(256).to(torch.uint8)
    frame = frame.reshape(3, 20, 30)
    images = torch.stack([frame, frame.clone()], dim=0)
    augment = PairedAugment(
        PairedAugmentConfig(
            enabled=True,
            horizontal_flip_p=1.0,
            crop_scale=(0.8, 0.8),
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            gamma=(0.9, 1.1),
            color_shared=True,
            noise_std=0.0,
            erase_p=1.0,
            erase_scale=(0.05, 0.05),
        )
    )
    output = augment(images)
    assert output.dtype == torch.uint8
    assert output.shape == images.shape
    assert torch.equal(output[0], output[1])


def test_focal_gamma_zero_matches_weighted_cross_entropy():
    logits = torch.tensor([[1.2, -0.2], [0.1, 1.1], [-0.4, 0.7]])
    targets = torch.tensor([0, 1, 1])
    alpha = [0.8, 1.3]
    focal = FocalLoss(gamma=0.0, alpha=alpha)(logits, targets)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(alpha))(logits, targets)
    assert torch.allclose(focal, ce, atol=1e-7, rtol=1e-6)
