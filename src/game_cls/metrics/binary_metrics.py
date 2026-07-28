from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from game_cls.losses.threshold_loss import probability_threshold_to_margin


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float

    def to_dict(self) -> dict:
        return asdict(self)


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _roc_auc(scores: list[float], targets: list[int]) -> float:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        return 0.0
    ranked = sorted(zip(scores, targets))
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            target for _, target in ranked[start:end]
        )
        start = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _average_precision(scores: list[float], targets: list[int]) -> float:
    positives = sum(targets)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, targets), reverse=True)
    tp = fp = 0
    area = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        previous_recall = tp / positives
        group_targets = [target for _, target in ranked[start:end]]
        tp += sum(group_targets)
        fp += len(group_targets) - sum(group_targets)
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        start = end
    return area


def confusion_from_margins(
    margins: Iterable[float], targets: Iterable[int], threshold: float = 0.99
) -> BinaryMetrics:
    margins = [float(value) for value in margins]
    targets = [int(value) for value in targets]
    cutoff = probability_threshold_to_margin(threshold)
    tp = fp = fn = tn = 0
    for margin, target in zip(margins, targets):
        prediction = float(margin) > cutoff
        if prediction and target == 1:
            tp += 1
        elif prediction:
            fp += 1
        elif target == 1:
            fn += 1
        else:
            tn += 1
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    return BinaryMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=_divide(2 * precision * recall, precision + recall),
        accuracy=_divide(tp + tn, tp + fp + fn + tn),
        specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2,
        roc_auc=_roc_auc(margins, targets),
        pr_auc=_average_precision(margins, targets),
    )


def probability_from_margin(margin: float) -> float:
    if margin >= 0:
        z = math.exp(-margin)
        return 1 / (1 + z)
    z = math.exp(margin)
    return z / (1 + z)
