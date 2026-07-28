from __future__ import annotations

import math
import unittest

from game_cls.losses.threshold_loss import (
    probability_threshold_to_margin,
    threshold_weight_at_step,
)
from game_cls.metrics.binary_metrics import confusion_from_margins


class ThresholdTests(unittest.TestCase):
    def test_point_99_is_log_99(self) -> None:
        self.assertAlmostEqual(probability_threshold_to_margin(0.99), math.log(99))

    def test_strict_greater_than_threshold(self) -> None:
        cutoff = probability_threshold_to_margin(0.99)
        metrics = confusion_from_margins(
            [cutoff, cutoff + 1e-6, -1.0, 10.0], [1, 1, 0, 0]
        )
        self.assertEqual((metrics.tp, metrics.fp, metrics.fn, metrics.tn), (1, 1, 1, 1))

    def test_threshold_weight_schedule(self) -> None:
        args = dict(
            total_steps=100,
            max_weight=0.2,
            warmup_ratio=0.1,
            ramp_ratio=0.2,
        )
        self.assertEqual(threshold_weight_at_step(10, **args), 0.0)
        self.assertAlmostEqual(threshold_weight_at_step(20, **args), 0.1)
        self.assertAlmostEqual(threshold_weight_at_step(30, **args), 0.2)
        self.assertAlmostEqual(threshold_weight_at_step(100, **args), 0.2)

    def test_auc_ties_are_not_order_dependent(self) -> None:
        metrics = confusion_from_margins([0.0, 0.0], [0, 1])
        self.assertAlmostEqual(metrics.roc_auc, 0.5)
        self.assertAlmostEqual(metrics.pr_auc, 0.5)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the current interpreter")
class TorchThresholdLossTests(unittest.TestCase):
    def test_loss_direction_and_finite_gradient(self) -> None:
        from game_cls.losses.threshold_loss import threshold_margin_loss

        logits = torch.tensor(
            [[0.0, 0.0], [0.0, 5.0], [0.0, 4.6], [0.0, -1.0]],
            requires_grad=True,
        )
        targets = torch.tensor([1, 1, 0, 0])
        loss = threshold_margin_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
