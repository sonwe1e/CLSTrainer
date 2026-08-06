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

    def test_negative_tail_ohem_picks_hardest(self) -> None:
        from game_cls.losses.threshold_loss import negative_tail_ohem_loss

        logits = torch.tensor([[0.0, 0.0], [0.0, 10.0], [0.0, 4.5], [0.0, 8.0]])
        targets = torch.tensor([0, 0, 1, 0])  # margins: 0.0, 10.0, 4.5(pos), 8.0
        full = negative_tail_ohem_loss(logits, targets)
        top1 = negative_tail_ohem_loss(logits, targets, hard_negative_k=1)
        # k >= number of negatives degenerates to the plain mean.
        top_all = negative_tail_ohem_loss(logits, targets, hard_negative_k=3)
        self.assertAlmostEqual(top_all.item(), full.item())
        self.assertGreater(top1.item(), full.item())
        margin = logits[:, 1] - logits[:, 0]
        neg_margin = margin[targets < 0.5]
        neg_loss = 0.5 * torch.nn.functional.softplus(
            (neg_margin - math.log(99) + 0.2) / 0.5
        )
        self.assertAlmostEqual(top1.item(), neg_loss.max().item())

    def test_negative_tail_ohem_no_negatives_is_zero(self) -> None:
        from game_cls.losses.threshold_loss import negative_tail_ohem_loss

        logits = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
        targets = torch.tensor([1, 1])
        loss = negative_tail_ohem_loss(logits, targets, hard_negative_k=2)
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_pairwise_ranking_penalty(self) -> None:
        from game_cls.losses.threshold_loss import pairwise_ranking_loss

        # Positive margin 4.0 vs hard negative margin 4.5 (>= log(99) - 0.2):
        # relu(0.2 - 4.0 + 4.5) = 0.7.
        logits = torch.tensor([[0.0, 4.0], [0.0, 4.5]])
        targets = torch.tensor([1, 0])
        loss = pairwise_ranking_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), 0.7)

        # Positive already at least rank_margin above the hard negative -> 0.
        well_separated = torch.tensor([[0.0, 5.0], [0.0, 4.5]])
        self.assertEqual(
            pairwise_ranking_loss(well_separated, torch.tensor([1, 0])).item(),
            0.0,
        )

        # No negatives (or no positives) -> detached zero, no NaN.
        all_positive = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        empty = pairwise_ranking_loss(all_positive, torch.ones(2))
        self.assertEqual(empty.item(), 0.0)
        self.assertFalse(torch.isnan(empty))

    def test_negative_tail_ohem_selects_top_k(self) -> None:
        from game_cls.losses.threshold_loss import negative_tail_ohem_loss

        # Six negatives with distinct margins; k=3 < 6 must select the three
        # hardest (largest individual losses) and average exactly those.
        logits = torch.tensor(
            [
                [0.0, -2.0],
                [0.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
                [0.0, 6.0],
                [0.0, 8.0],
            ]
        )
        targets = torch.zeros(6)
        margin = logits[:, 1] - logits[:, 0]
        per_sample = 0.5 * torch.nn.functional.softplus(
            (margin - math.log(99) + 0.2) / 0.5
        )
        top3 = torch.topk(per_sample, 3).values.mean()
        self.assertAlmostEqual(
            negative_tail_ohem_loss(logits, targets, hard_negative_k=3).item(),
            top3.item(),
        )
        # A subset of the hardest negatives always exceeds the full mean.
        self.assertGreater(
            negative_tail_ohem_loss(logits, targets, hard_negative_k=2).item(),
            negative_tail_ohem_loss(logits, targets).item(),
        )

    def test_pairwise_ranking_direction(self) -> None:
        from game_cls.losses.threshold_loss import pairwise_ranking_loss

        # Fixed hard negatives (margins 4.5/4.6, at/above log(99)-0.2); raising
        # the positive margin strictly lowers the loss.
        hard_negatives = torch.tensor([[0.0, 4.5], [0.0, 4.6]])
        low = pairwise_ranking_loss(
            torch.cat([torch.tensor([[0.0, 4.0]]), hard_negatives]),
            torch.tensor([1, 0, 0]),
        )
        high = pairwise_ranking_loss(
            torch.cat([torch.tensor([[0.0, 4.7]]), hard_negatives]),
            torch.tensor([1, 0, 0]),
        )
        self.assertLess(high.item(), low.item())
        # rank_margin satisfied: the positive sits at least margin above every
        # hard negative, so the penalty is exactly zero.
        satisfied = pairwise_ranking_loss(
            torch.cat([torch.tensor([[0.0, 5.0]]), hard_negatives]),
            torch.tensor([1, 0, 0]),
        )
        self.assertEqual(satisfied.item(), 0.0)
        # Zero scalar when either class is absent (no NaN).
        only_pos = pairwise_ranking_loss(
            torch.tensor([[0.0, 4.0], [0.0, 4.5]]), torch.ones(2)
        )
        only_neg = pairwise_ranking_loss(
            torch.tensor([[0.0, 4.0], [0.0, 4.5]]), torch.zeros(2)
        )
        self.assertEqual(only_pos.item(), 0.0)
        self.assertEqual(only_neg.item(), 0.0)
        self.assertFalse(torch.isnan(only_pos))
        self.assertFalse(torch.isnan(only_neg))

    def test_combined_loss_zero_weights_matches_default(self) -> None:
        from torch.nn import functional as F

        from game_cls.losses.threshold_loss import (
            combined_loss,
            threshold_margin_loss,
            threshold_weight_at_step,
        )

        logits = torch.tensor([[0.0, 0.0], [0.0, 5.0], [0.0, 4.6], [0.0, -1.0]])
        targets = torch.tensor([1, 1, 0, 0])
        base = {
            "cross_entropy_weight": 1.0,
            "threshold": 0.99,
            "threshold_loss_weight": 0.2,
            "threshold_safety_margin": 0.2,
            "threshold_temperature": 0.5,
        }
        total_default, comp_default = combined_loss(
            logits, targets, config=base, step=50, total_steps=100
        )
        # The same base with the stage-4 keys explicitly zeroed must be
        # numerically identical to a config that never mentions them.
        zeroed = {
            **base,
            "negative_tail_loss_weight": 0.0,
            "negative_tail_hard_negative_k": None,
            "rank_loss_weight": 0.0,
            "rank_margin": 0.2,
        }
        total, components = combined_loss(
            logits, targets, config=zeroed, step=50, total_steps=100
        )
        self.assertAlmostEqual(total.item(), total_default.item())
        self.assertEqual(components["negative_tail_loss"].item(), 0.0)
        self.assertEqual(components["rank_loss"].item(), 0.0)
        self.assertEqual(set(comp_default), set(components))
        # Hand-replicated legacy baseline: ce + weight * margin loss.
        weight = threshold_weight_at_step(50, 100, 0.2, 0.10, 0.20)
        expected = F.cross_entropy(
            logits.float(), targets
        ) + weight * threshold_margin_loss(logits, targets)
        self.assertAlmostEqual(total_default.item(), expected.item())

    def test_combined_loss_enabling_terms_changes_total(self) -> None:
        from game_cls.losses.threshold_loss import combined_loss

        # Hard negatives (margins 4.6, 8.0) sit at/above the decision boundary
        # against positives at 4.0/4.5, so both new terms are nonzero.
        logits = torch.tensor([[0.0, 4.0], [0.0, 4.5], [0.0, 4.6], [0.0, 8.0]])
        targets = torch.tensor([1, 1, 0, 0])
        base = {"threshold_loss_weight": 0.0}  # isolate the stage-4 terms
        total_off, comp_off = combined_loss(
            logits, targets, config=base, step=0, total_steps=10
        )
        self.assertEqual(comp_off["negative_tail_loss"].item(), 0.0)
        self.assertEqual(comp_off["rank_loss"].item(), 0.0)
        on = {
            **base,
            "negative_tail_loss_weight": 0.5,
            "negative_tail_hard_negative_k": 1,
            "rank_loss_weight": 0.5,
            "rank_margin": 0.2,
        }
        total_on, comp_on = combined_loss(
            logits, targets, config=on, step=0, total_steps=10
        )
        self.assertGreater(comp_on["negative_tail_loss"].item(), 0.0)
        self.assertGreater(comp_on["rank_loss"].item(), 0.0)
        expected = (
            total_off + 0.5 * comp_on["negative_tail_loss"] + 0.5 * comp_on["rank_loss"]
        )
        self.assertAlmostEqual(total_on.item(), expected.item())
        # Enabling the terms raises the total for this adversarial batch.
        self.assertGreater(total_on.item(), total_off.item())

    def test_combined_loss_defaults_include_optional_keys(self) -> None:
        from game_cls.losses.threshold_loss import combined_loss

        logits = torch.tensor([[0.0, 0.0], [0.0, 5.0], [0.0, 4.6], [0.0, -1.0]])
        targets = torch.tensor([1, 1, 0, 0])
        total, components = combined_loss(
            logits, targets, config={}, step=100, total_steps=100
        )
        self.assertEqual(
            set(components),
            {
                "cross_entropy",
                "threshold_loss",
                "threshold_weight",
                "negative_tail_loss",
                "rank_loss",
            },
        )
        self.assertEqual(components["negative_tail_loss"].item(), 0.0)
        self.assertEqual(components["rank_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
