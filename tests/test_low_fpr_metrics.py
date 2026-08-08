"""Low-FPR evaluation protocol (step4 §三.3).

Runs ``evaluate()`` on small CPU synthetic datasets and checks the eleven
new metric keys: consistency of the decision-threshold confusion matrix
against a manual recomputation, the threshold-independent low-FPR metrics,
the tail-calibration gate and the worst-game keys.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from game_cls.data.collate import pair_collate
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.training.synthetic import SyntheticPairDataset
from game_cls.losses.threshold_loss import probability_threshold_to_margin

NEW_METRIC_KEYS = (
    "global_fpr_at_decision_threshold",
    "global_specificity_at_decision_threshold",
    "global_positive_recall_at_decision_threshold",
    "worst_game_fpr_at_decision_threshold",
    "worst_game_positive_recall_at_decision_threshold",
    "negative_score_p99",
    "negative_score_p999",
    "negative_score_max",
    "recall_at_max_fpr",
    "low_fpr_partial_auc",
    "ece_tail_95_100",
)


class SeparatingModel(torch.nn.Module):
    """Scores the image0 top/bottom channel-0 mean difference.

    Positive training pairs have +128 added to the top half of channel 0, so
    this margin separates positives (diff ~= 0.5) from negatives (diff ~= 0)
    with a large gap.
    """

    def forward(self, image0, image1):
        top = image0[:, 0, : image0.shape[2] // 2].mean(dim=(1, 2))
        bottom = image0[:, 0, image0.shape[2] // 2 :].mean(dim=(1, 2))
        score = (top - bottom) * 150.0 - 60.0
        return torch.stack((-score, score), dim=1)


class ControlledDataset:
    """Deterministic tiny dataset with hand-picked top-half brightness.

    Each sample is an 8x8 pair; the top half of image0 channel 0 is set to
    ``value``, which the SeparatingModel turns into a known margin:
    value 255 -> TP, 106 -> TP, 105 -> FN, 107 -> FP, 0 -> TN.
    """

    samples = [
        (255, 1),
        (106, 1),
        (105, 1),
        (107, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        value, label = self.samples[index]
        images = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
        images[0, 0, :4, :] = value
        return {
            "images": images,
            "label": label,
            "meta": {
                "game": "A" if index % 2 == 0 else "B",
                "video_id": f"{index:02d}",
            },
        }


def _manual_confusion(model, dataset, threshold: float) -> tuple[int, int, int, int]:
    """Recompute the decision-threshold confusion matrix by hand."""
    cutoff = probability_threshold_to_margin(threshold)
    tp = fp = fn = tn = 0
    for index in range(len(dataset)):
        item = dataset[index]
        images = item["images"].unsqueeze(0).float().div_(255.0)
        label = int(item["label"])
        with torch.no_grad():
            logits = model(images[:, 0], images[:, 1])
        margin = float(logits[0, 1] - logits[0, 0])
        prediction = int(margin > cutoff)
        if prediction and label == 1:
            tp += 1
        elif prediction and label == 0:
            fp += 1
        elif not prediction and label == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


@unittest.skipIf(torch is None, "torch is not installed")
class LowFprEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.synthetic = SyntheticPairDataset(
            16, ImageSpec(width=16, height=32, channels=3), 77
        )

    def _loader(self, dataset):
        from torch.utils.data import DataLoader

        return DataLoader(dataset, batch_size=4, collate_fn=pair_collate)

    def test_evaluator_emits_all_low_fpr_metric_keys(self) -> None:
        from game_cls.engine.evaluator import evaluate

        result = evaluate(
            SeparatingModel(),
            self._loader(self.synthetic),
            self.device,
        )
        for key in NEW_METRIC_KEYS:
            self.assertIn(key, result.metrics, key)

    def test_confusion_matrix_matches_manual_recompute(self) -> None:
        from game_cls.engine.evaluator import evaluate

        dataset = ControlledDataset()
        result = evaluate(SeparatingModel(), self._loader(dataset), self.device)
        tp, fp, fn, tn = _manual_confusion(SeparatingModel(), dataset, 0.99)
        self.assertEqual((tp, fp, fn, tn), (2, 1, 1, 3))
        self.assertAlmostEqual(
            result.metrics["global_fpr_at_decision_threshold"],
            fp / (fp + tn),
        )
        self.assertAlmostEqual(
            result.metrics["global_specificity_at_decision_threshold"],
            tn / (fp + tn),
        )
        self.assertAlmostEqual(
            result.metrics["global_positive_recall_at_decision_threshold"],
            tp / (tp + fn),
        )
        # Non-trivial counts: the evaluator must see both errors and passes.
        self.assertEqual(
            (
                result.metrics["global_fpr_at_decision_threshold"],
                result.metrics["global_positive_recall_at_decision_threshold"],
            ),
            (0.25, 2 / 3),
        )

    def test_perfect_model_reaches_full_recall_and_partial_auc(self) -> None:
        from game_cls.engine.evaluator import evaluate

        result = evaluate(
            SeparatingModel(),
            self._loader(self.synthetic),
            self.device,
        )
        recall_at_max_fpr = result.metrics["recall_at_max_fpr"]
        partial_auc = result.metrics["low_fpr_partial_auc"]
        self.assertGreaterEqual(recall_at_max_fpr, 0.0)
        self.assertLessEqual(recall_at_max_fpr, 1.0)
        self.assertGreaterEqual(partial_auc, 0.0)
        self.assertLessEqual(partial_auc, 1.0)
        # A perfect separator: recall reaches 1.0 before the FPR bound, so
        # both threshold-independent metrics score exactly 1.0.
        self.assertEqual(recall_at_max_fpr, 1.0)
        self.assertEqual(partial_auc, 1.0)

    def test_low_fpr_metrics_are_bounded_for_imperfect_model(self) -> None:
        from game_cls.engine.evaluator import evaluate

        result = evaluate(
            SeparatingModel(),
            self._loader(ControlledDataset()),
            self.device,
        )
        for key in ("recall_at_max_fpr", "low_fpr_partial_auc"):
            value = result.metrics[key]
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_tail_calibration_is_finite_and_gated(self) -> None:
        from game_cls.engine.evaluator import evaluate

        loader = self._loader(ControlledDataset())
        enabled = evaluate(
            SeparatingModel(),
            loader,
            self.device,
            tail_calibration_enabled=True,
        )
        ece = enabled.metrics["ece_tail_95_100"]
        self.assertTrue(torch.isfinite(torch.tensor(ece)))
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertGreater(ece, 0.0)  # the controlled set is miscalibrated
        disabled = evaluate(
            SeparatingModel(),
            loader,
            self.device,
            tail_calibration_enabled=False,
        )
        self.assertEqual(disabled.metrics["ece_tail_95_100"], 0.0)

    def test_worst_game_metrics_are_finite(self) -> None:
        from game_cls.engine.evaluator import evaluate

        result = evaluate(
            SeparatingModel(),
            self._loader(ControlledDataset()),
            self.device,
        )
        worst_fpr = result.metrics["worst_game_fpr_at_decision_threshold"]
        worst_recall = result.metrics[
            "worst_game_positive_recall_at_decision_threshold"
        ]
        for value in (worst_fpr, worst_recall):
            self.assertTrue(torch.isfinite(torch.tensor(value)))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        # Two games with non-trivial worst values (A: fpr 0 / recall 0.5,
        # B: fpr 0.5 / recall 1.0).
        self.assertEqual((worst_fpr, worst_recall), (0.5, 0.5))

    def test_max_fpr_for_recall_is_honoured(self) -> None:
        from game_cls.engine.evaluator import evaluate

        loader = self._loader(ControlledDataset())
        default = evaluate(
            SeparatingModel(),
            loader,
            self.device,
            max_fpr_for_recall=0.01,
        )
        wider = evaluate(
            SeparatingModel(),
            loader,
            self.device,
            max_fpr_for_recall=0.5,
        )
        # A wider FPR bound can only keep or raise recall-at-bound.
        self.assertGreaterEqual(
            wider.metrics["recall_at_max_fpr"],
            default.metrics["recall_at_max_fpr"],
        )


if __name__ == "__main__":
    unittest.main()
