from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist


OUTCOMES = ("tp", "fp", "fn", "tn")


def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    samples = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / samples if samples else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "samples": samples,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _per_class(tp: int, fp: int, fn: int, tn: int) -> dict[str, dict[str, float | int]]:
    # Treat each class as the positive class in turn.
    positive = _binary_metrics(tp, fp, fn, tn)
    negative = _binary_metrics(tn, fn, fp, tp)
    return {
        "0": {
            "precision": negative["precision"],
            "recall": negative["recall"],
            "f1": negative["f1"],
            "support": tn + fp,
        },
        "1": {
            "precision": positive["precision"],
            "recall": positive["recall"],
            "f1": positive["f1"],
            "support": tp + fn,
        },
    }


def _hist_quantile(hist: Sequence[int], q: float) -> float | None:
    total = sum(int(x) for x in hist)
    if total <= 0:
        return None
    target = max(1, int(round(float(q) * total)))
    running = 0
    bins = len(hist)
    for index, value in enumerate(hist):
        running += int(value)
        if running >= target:
            return (index + 0.5) / bins
    return (bins - 0.5) / bins


def _confidence_summary(count: int, value_sum: float, hist: Sequence[int]) -> dict:
    return {
        "count": int(count),
        "mean": value_sum / count if count else None,
        "p10": _hist_quantile(hist, 0.10),
        "p50": _hist_quantile(hist, 0.50),
        "p90": _hist_quantile(hist, 0.90),
    }


@dataclass
class BinaryAccumulator:
    """Accumulate loss/confusion counts on-device; sync only for logs/end-of-epoch."""

    device: torch.device

    def __post_init__(self) -> None:
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.samples = torch.zeros((), dtype=torch.int64, device=self.device)
        self.counts = torch.zeros(4, dtype=torch.int64, device=self.device)

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> None:
        batch_size = targets.numel()
        self.loss_sum += loss.detach().float() * batch_size
        self.samples += batch_size

        probability = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        prediction = probability >= float(threshold)
        positive = targets.detach() == 1
        self.counts[0] += (prediction & positive).sum()
        self.counts[1] += (prediction & ~positive).sum()
        self.counts[2] += (~prediction & positive).sum()
        self.counts[3] += (~prediction & ~positive).sum()

    def reduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.counts, op=dist.ReduceOp.SUM)

    def compute(self) -> dict[str, float | int]:
        loss_sum = float(self.loss_sum.item())
        samples = int(self.samples.item())
        tp, fp, fn, tn = (int(x) for x in self.counts.tolist())
        result = _binary_metrics(tp, fp, fn, tn)
        result["loss"] = loss_sum / samples if samples else 0.0
        return result


class DiagnosticAccumulator:
    """DDP-safe grouped evaluator without retaining per-sample predictions.

    Counts and small histograms stay on the accelerator and are all-reduced at
    the end. This gives per-game/per-class confusion plus confidence diagnostics
    while keeping run artifacts tiny even for large validation/test sets.
    """

    def __init__(self, device: torch.device, games: Sequence[str], *, bins: int = 20) -> None:
        if bins < 5:
            raise ValueError("diagnostic bins must be >= 5")
        self.device = device
        self.games = tuple(sorted(set(str(game) for game in games)))
        if not self.games:
            raise ValueError("diagnostics require at least one game")
        self.game_to_index = {game: index for index, game in enumerate(self.games)}
        self.bins = int(bins)
        self.threshold: float | None = None
        game_count = len(self.games)
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        self.samples = torch.zeros((), dtype=torch.int64, device=device)
        self.counts = torch.zeros((game_count, 4), dtype=torch.int64, device=device)
        self.confidence_sum = torch.zeros((game_count, 4), dtype=torch.float32, device=device)
        self.confidence_hist = torch.zeros(
            (game_count, 4, self.bins), dtype=torch.int64, device=device
        )
        self.score_hist = torch.zeros(
            (game_count, 2, self.bins), dtype=torch.int64, device=device
        )

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        games: Sequence[str],
        threshold: float,
    ) -> None:
        batch_size = targets.numel()
        if len(games) != batch_size:
            raise ValueError("game metadata length does not match batch size")
        self.loss_sum += loss.detach().float() * batch_size
        self.samples += batch_size
        if self.threshold is None:
            self.threshold = float(threshold)
        elif abs(self.threshold - float(threshold)) > 1e-12:
            raise ValueError("diagnostic threshold changed within one evaluation")

        probability = torch.softmax(logits.detach().float(), dim=1)[:, 1]
        prediction = probability >= float(threshold)
        positive = targets.detach() == 1
        outcome = torch.empty_like(targets, dtype=torch.long)
        outcome[prediction & positive] = 0
        outcome[prediction & ~positive] = 1
        outcome[~prediction & positive] = 2
        outcome[~prediction & ~positive] = 3
        game_index = torch.tensor(
            [self.game_to_index[str(game)] for game in games],
            dtype=torch.long,
            device=self.device,
        )

        flat_outcome = game_index * 4 + outcome
        ones = torch.ones(batch_size, dtype=torch.int64, device=self.device)
        self.counts.view(-1).scatter_add_(0, flat_outcome, ones)

        confidence = torch.where(prediction, probability, 1.0 - probability)
        self.confidence_sum.view(-1).scatter_add_(0, flat_outcome, confidence)
        confidence_bin = torch.clamp((confidence * self.bins).long(), 0, self.bins - 1)
        confidence_key = flat_outcome * self.bins + confidence_bin
        self.confidence_hist.view(-1).scatter_add_(0, confidence_key, ones)

        score_bin = torch.clamp((probability * self.bins).long(), 0, self.bins - 1)
        score_key = (game_index * 2 + targets.long()) * self.bins + score_bin
        self.score_hist.view(-1).scatter_add_(0, score_key, ones)

    def reduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        for tensor in (
            self.loss_sum,
            self.samples,
            self.counts,
            self.confidence_sum,
            self.confidence_hist,
            self.score_hist,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def _game_report(self, game_index: int) -> dict:
        counts = [int(x) for x in self.counts[game_index].tolist()]
        tp, fp, fn, tn = counts
        report = _binary_metrics(tp, fp, fn, tn)
        report["per_class"] = _per_class(tp, fp, fn, tn)
        report["confidence"] = {
            name: _confidence_summary(
                counts[outcome_index],
                float(self.confidence_sum[game_index, outcome_index].item()),
                [int(x) for x in self.confidence_hist[game_index, outcome_index].tolist()],
            )
            for outcome_index, name in enumerate(OUTCOMES)
        }
        report["score_histogram"] = {
            "class0": [int(x) for x in self.score_hist[game_index, 0].tolist()],
            "class1": [int(x) for x in self.score_hist[game_index, 1].tolist()],
        }
        return report

    def compute(self) -> dict:
        samples = int(self.samples.item())
        total_counts = self.counts.sum(dim=0)
        tp, fp, fn, tn = (int(x) for x in total_counts.tolist())
        result = _binary_metrics(tp, fp, fn, tn)
        result["loss"] = float(self.loss_sum.item()) / samples if samples else 0.0
        result["threshold"] = self.threshold
        result["per_class"] = _per_class(tp, fp, fn, tn)
        result["per_game"] = {
            game: self._game_report(index) for index, game in enumerate(self.games)
        }

        confidence_counts = self.counts.sum(dim=0)
        confidence_sums = self.confidence_sum.sum(dim=0)
        confidence_hist = self.confidence_hist.sum(dim=0)
        result["confidence"] = {
            name: _confidence_summary(
                int(confidence_counts[index].item()),
                float(confidence_sums[index].item()),
                [int(x) for x in confidence_hist[index].tolist()],
            )
            for index, name in enumerate(OUTCOMES)
        }
        score_hist = self.score_hist.sum(dim=0)
        result["score_histogram"] = {
            "bins": self.bins,
            "class0": [int(x) for x in score_hist[0].tolist()],
            "class1": [int(x) for x in score_hist[1].tolist()],
        }
        return result
