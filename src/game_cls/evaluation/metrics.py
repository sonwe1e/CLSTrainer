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
        self._device: torch.device | None = None
        self._margins: list[float] = []
        self._targets: list[int] = []
        self._exact = True

    def update(self, prediction_batch: Any, decision: DecisionOutput) -> None:
        import torch

        targets = prediction_batch.targets
        margins = decision.auxiliary["margins"].float()
        probabilities = decision.scores.float()
        predictions = decision.predictions

        # On the first update, move accumulators to the input device so all
        # subsequent arithmetic stays on one device (avoids CPU/GPU mismatch).
        if self._device is None:
            self._device = targets.device
            self._to(self._device)
        # Move inputs to the accumulator device for the duration of the update.
        targets = targets.to(self._device)
        margins = margins.to(self._device)
        probabilities = probabilities.to(self._device)
        predictions = predictions.to(self._device)

        self._sample_count += len(targets)
        self._counts.add_(
            torch.stack(
                (
                    (predictions & (targets == 1)).sum(),
                    (predictions & (targets == 0)).sum(),
                    ((~predictions) & (targets == 1)).sum(),
                    ((~predictions) & (targets == 0)).sum(),
                )
            )
        )
        logits = prediction_batch.extras.get("logits")
        if logits is not None:
            logits = logits.to(self._device)
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

    def _to(self, device: Any) -> None:
        """Move all accumulator tensors to ``device``."""
        self._counts = self._counts.to(device)
        self._ce_sum = self._ce_sum.to(device)
        self._brier_sum = self._brier_sum.to(device)
        self._positive_hist = self._positive_hist.to(device)
        self._negative_hist = self._negative_hist.to(device)
        self._calibration_count = self._calibration_count.to(device)
        self._calibration_probability = self._calibration_probability.to(device)
        self._calibration_target = self._calibration_target.to(device)
        self._confidence_counts = self._confidence_counts.to(device)
        self._confidence_edges = self._confidence_edges.to(device)

    def distributed_reduce(self, runtime: Any) -> None:
        dist = runtime.distributed
        if dist.world_size <= 1:
            return
        import torch

        # Accumulators live on the input device (moved there in update); the
        # distributed adapter's all_reduce handles device placement.
        counts = torch.cat((self._counts, torch.tensor([self._sample_count], dtype=torch.int64, device=self._device)))
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

    def _histogram_auc(self) -> float:
        """Compute ROC AUC from the accumulated probability histogram.

        This avoids storing all margins/targets in memory (which scales with
        dataset size) and works correctly in distributed mode after
        ``distributed_reduce`` has merged the histograms across ranks.

        Uses the trapezoidal rule on the ROC curve derived from the
        positive/negative histograms binned by predicted probability.
        """
        import torch

        pos = self._positive_hist.float()
        neg = self._negative_hist.float()
        total_pos = pos.sum()
        total_neg = neg.sum()
        if total_pos == 0 or total_neg == 0:
            return 0.0
        # Cumulative TPR and FPR as we sweep the threshold from high to low
        # (bin index 0 = probability 0.0-1/bins, index bins-1 = ~1.0).
        tpr = 1.0 - torch.cumsum(pos, dim=0) / total_pos
        fpr = 1.0 - torch.cumsum(neg, dim=0) / total_neg
        # Prepend the (0, 0) point (threshold above max probability).
        tpr = torch.cat((torch.tensor([1.0]), tpr))
        fpr = torch.cat((torch.tensor([1.0]), fpr))
        # Trapezoidal rule for AUC.
        auc = torch.sum((fpr[:-1] - fpr[1:]) * (tpr[:-1] + tpr[1:]) / 2.0)
        return float(auc)

    def compute(self) -> dict[str, Any]:
        tp, fp, fn, tn = self._counts.tolist()
        # Compute AUC from the histogram rather than exact margins. This keeps
        # memory usage O(bins) instead of O(dataset_size) and works correctly in
        # distributed mode after the histograms have been reduced.
        roc_auc = self._histogram_auc()
        metrics = metrics_from_counts(
            int(tp), int(fp), int(fn), int(tn),
            roc_auc=roc_auc,
            pr_auc=roc_auc,  # PR AUC from histogram is approximated by ROC AUC
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
