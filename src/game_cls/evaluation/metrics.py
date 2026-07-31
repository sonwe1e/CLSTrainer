from __future__ import annotations

from typing import Any

from ..contracts.evaluation import DecisionOutput, MetricAccumulator
from ..metrics.binary_metrics import confusion_from_margins, metrics_from_counts


class BinaryConfusionAccumulator(MetricAccumulator):
    """Accumulates the global binary confusion matrix, CE, Brier, AUC, calibration.

    This is the faithful extraction of the metric accumulation that previously
    lived inside ``evaluate`` (USERPLAN §10.3). It reduces across ranks and
    computes the final metric dict via the existing ``binary_metrics`` helpers.
    """

    def __init__(self, auc_histogram_bins: int = 4096) -> None:
        self.auc_histogram_bins = auc_histogram_bins
        import torch

        self._counts = torch.zeros(4, dtype=torch.int64)
        self._sample_count = 0
        self._ce_sum = torch.zeros((), dtype=torch.float32)
        self._brier_sum = torch.zeros((), dtype=torch.float32)
        self._positive_hist = torch.zeros(auc_histogram_bins, dtype=torch.int64)
        self._negative_hist = torch.zeros(auc_histogram_bins, dtype=torch.int64)
        self._calibration_count = torch.zeros(20, dtype=torch.int64)
        self._calibration_probability = torch.zeros(20, dtype=torch.float32)
        self._calibration_target = torch.zeros(20, dtype=torch.int64)
        self._confidence_counts = torch.zeros(5, dtype=torch.int64)
        self._confidence_edges = torch.tensor(
            [0.980, 0.990, 0.995, 0.999], dtype=torch.float32
        )
        self._margins: list[float] = []
        self._targets: list[int] = []
        self._exact = True

    def update(self, prediction_batch: Any, decision: DecisionOutput) -> None:
        import torch

        targets = prediction_batch.targets
        margins = decision.auxiliary["margins"].float()
        probabilities = decision.scores.float()
        predictions = decision.predictions

        self._sample_count += len(targets)
        self._counts.add_(
            torch.stack(
                (
                    (predictions & (targets == 1)).sum(),
                    (predictions & (targets == 0)).sum(),
                    ((~predictions) & (targets == 1)).sum(),
                    ((~predictions) & (targets == 0)).sum(),
                )
            ).to(torch.int64)
        )
        logits = prediction_batch.extras.get("logits")
        if logits is not None:
            self._ce_sum.add_(
                torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
            )
            self._brier_sum.add_(torch.square(probabilities - targets.float()).sum())

        auc_indices = torch.clamp(
            (probabilities * self.auc_histogram_bins).long(),
            max=self.auc_histogram_bins - 1,
        )
        self._positive_hist += torch.bincount(
            auc_indices[targets == 1], minlength=self.auc_histogram_bins
        )
        self._negative_hist += torch.bincount(
            auc_indices[targets == 0], minlength=self.auc_histogram_bins
        )
        calibration_indices = torch.clamp((probabilities * 20).long(), max=19)
        self._calibration_count += torch.bincount(calibration_indices, minlength=20)
        self._calibration_probability.scatter_add_(0, calibration_indices, probabilities)
        self._calibration_target.scatter_add_(0, calibration_indices, targets.to(torch.int64))
        confidence_indices = sum(
            probabilities >= edge for edge in self._confidence_edges
        ).to(torch.int64)
        self._confidence_counts += torch.bincount(confidence_indices, minlength=5)

        self._margins.extend(margins.cpu().tolist())
        self._targets.extend(targets.cpu().tolist())

    def distributed_reduce(self, runtime: Any) -> None:
        import torch

        dist = runtime.distributed
        if dist.world_size <= 1:
            return
        counts = torch.cat((self._counts, torch.tensor([self._sample_count], dtype=torch.int64)))
        floating = torch.stack((self._ce_sum, self._brier_sum))
        dist.all_reduce(counts)
        dist.all_reduce(floating)
        dist.all_reduce(self._positive_hist)
        dist.all_reduce(self._negative_hist)
        dist.all_reduce(self._calibration_count)
        dist.all_reduce(self._calibration_probability)
        dist.all_reduce(self._calibration_target)
        dist.all_reduce(self._confidence_counts)
        self._counts = counts[:4]
        self._sample_count = int(counts[4])
        self._ce_sum, self._brier_sum = floating[0], floating[1]

    def compute(self) -> dict[str, Any]:
        tp, fp, fn, tn = self._counts.tolist()
        metrics = metrics_from_counts(
            int(tp), int(fp), int(fn), int(tn),
            roc_auc=confusion_from_margins(self._margins, self._targets).roc_auc,
            pr_auc=confusion_from_margins(self._margins, self._targets).pr_auc,
        ).to_dict()
        sample_count = self._sample_count
        metrics.update(
            {
                "cross_entropy": self._ce_sum.item() / sample_count if sample_count else 0.0,
                "brier_score": self._brier_sum.item() / sample_count if sample_count else 0.0,
                "sample_count": sample_count,
                "confidence_histogram": dict(
                    zip(
                        ["<0.980", "0.980-0.990", "0.990-0.995", "0.995-0.999", ">=0.999"],
                        self._confidence_counts.tolist(),
                    )
                ),
            }
        )
        return metrics
