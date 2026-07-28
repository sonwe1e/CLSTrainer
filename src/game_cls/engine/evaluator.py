from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from game_cls.losses.threshold_loss import probability_threshold_to_margin
from game_cls.metrics.binary_metrics import (
    confusion_from_margins,
    metrics_from_counts,
    probability_from_margin,
)


@dataclass
class EvaluationOutput:
    metrics: dict | None
    grouped_metrics: dict | None
    errors: list[dict]
    near_threshold: list[dict]


def _metadata_dict(item: Any) -> dict:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return {"sample": str(item)}


def _threshold_band(probability: float) -> str | None:
    if 0.980 <= probability < 0.990:
        return "0.980-0.990"
    if 0.990 <= probability <= 0.995:
        return "0.990-0.995"
    return None


def _counter(margin: float, target: int, cutoff: float) -> list[int]:
    prediction = margin > cutoff
    return [
        int(prediction and target == 1),
        int(prediction and target == 0),
        int(not prediction and target == 1),
        int(not prediction and target == 0),
    ]


def _add_counter(target: dict, key, values: list[int]) -> None:
    current = target.setdefault(key, [0, 0, 0, 0])
    for index, value in enumerate(values):
        current[index] += value


def _merge_group_counters(items: list[dict]) -> dict:
    merged: dict = {}
    for item in items:
        for key, values in item.items():
            _add_counter(merged, key, values)
    return merged


def _group_rows(counters: dict, key_names: tuple[str, ...]) -> list[dict]:
    rows = []
    for key, counts in sorted(counters.items()):
        key = key if isinstance(key, tuple) else (key,)
        metrics = metrics_from_counts(*counts).to_dict()
        rows.append({**dict(zip(key_names, key)), **metrics})
    return rows


def _add_confidence(target: dict, key, probability: float) -> None:
    current = target.setdefault(
        key,
        {"count": 0, "sum": 0.0, "min": 1.0, "max": 0.0},
    )
    current["count"] += 1
    current["sum"] += probability
    current["min"] = min(current["min"], probability)
    current["max"] = max(current["max"], probability)


def _merge_confidence(items: list[dict]) -> dict:
    merged: dict = {}
    for item in items:
        for key, values in item.items():
            current = merged.setdefault(
                key,
                {"count": 0, "sum": 0.0, "min": 1.0, "max": 0.0},
            )
            current["count"] += values["count"]
            current["sum"] += values["sum"]
            current["min"] = min(current["min"], values["min"])
            current["max"] = max(current["max"], values["max"])
    return merged


def _video_rows(counters: dict, confidence: dict) -> list[dict]:
    rows = []
    for key, counts in sorted(counters.items()):
        game, label, video_id = key
        binary = metrics_from_counts(*counts).to_dict()
        values = confidence[key]
        row = {
            "game": game,
            "label": label,
            "video_id": video_id,
            **binary,
            "mean_probability_class1": values["sum"] / values["count"],
            "min_probability_class1": values["min"],
            "max_probability_class1": values["max"],
        }
        if label == 1:
            row.update(
                {
                    "primary_metric": "positive_recall",
                    "positive_recall": binary["recall"],
                    "fn_rate": 1.0 - binary["recall"],
                    "negative_specificity": None,
                    "fp_rate": None,
                }
            )
        else:
            row.update(
                {
                    "primary_metric": "negative_specificity",
                    "positive_recall": None,
                    "fn_rate": None,
                    "negative_specificity": binary["specificity"],
                    "fp_rate": 1.0 - binary["specificity"],
                }
            )
        rows.append(row)
    return rows


def _histogram_auc(
    positive_histogram: list[int], negative_histogram: list[int]
) -> tuple[float, float]:
    positives = sum(positive_histogram)
    negatives = sum(negative_histogram)
    roc_auc = 0.0
    if positives and negatives:
        negatives_below = 0
        concordant = 0.0
        for positive, negative in zip(positive_histogram, negative_histogram):
            concordant += positive * (negatives_below + 0.5 * negative)
            negatives_below += negative
        roc_auc = concordant / (positives * negatives)
    pr_auc = 0.0
    if positives:
        tp = fp = 0
        previous_recall = 0.0
        for positive, negative in zip(
            reversed(positive_histogram), reversed(negative_histogram)
        ):
            tp += positive
            fp += negative
            recall = tp / positives
            precision = tp / (tp + fp) if tp + fp else 0.0
            pr_auc += (recall - previous_recall) * precision
            previous_recall = recall
    return roc_auc, pr_auc


def _calibration_metrics(
    counts: list[int],
    probability_sums: list[float],
    target_sums: list[int],
    sample_count: int,
) -> float:
    if not sample_count:
        return 0.0
    ece = 0.0
    for count, probability_sum, target_sum in zip(
        counts, probability_sums, target_sums
    ):
        if count:
            confidence = probability_sum / count
            accuracy = target_sum / count
            ece += count / sample_count * abs(confidence - accuracy)
    return ece


def _make_reduction_tensors(local_counts, sample_count, cross_entropy_sum, brier_sum, device):
    import torch

    counts = torch.tensor(
        [*local_counts, sample_count],
        dtype=torch.int64,
        device=device,
    )
    floating = torch.tensor(
        [cross_entropy_sum, brier_sum],
        dtype=torch.float32,
        device=device,
    )
    return counts, floating


def evaluate(
    model,
    dataloader,
    device,
    threshold: float = 0.99,
    *,
    checkpoint_step: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    evaluation_kind: str = "quick",
    report_dir=None,
    full_auc_mode: str = "histogram",
    auc_histogram_bins: int = 4096,
    quick_error_limit: int = 200,
) -> EvaluationOutput:
    import torch

    from game_cls.reports.error_writer import EvaluationShardWriter

    model.eval()
    cutoff = probability_threshold_to_margin(threshold)
    exact_scores = evaluation_kind == "quick" or full_auc_mode == "exact"
    margins: list[float] = []
    targets: list[int] = []
    retained_errors: list[dict] = []
    retained_near: list[dict] = []
    group_game: dict = {}
    group_video: dict = {}
    group_game_label: dict = {}
    group_video_confidence: dict = {}
    cross_entropy_sum = 0.0
    brier_sum = 0.0
    sample_count = 0
    local_counts = [0, 0, 0, 0]
    positive_hist = torch.zeros(auc_histogram_bins, dtype=torch.int64, device=device)
    negative_hist = torch.zeros(auc_histogram_bins, dtype=torch.int64, device=device)
    calibration_count = torch.zeros(20, dtype=torch.int64, device=device)
    calibration_probability = torch.zeros(20, dtype=torch.float32, device=device)
    calibration_target = torch.zeros(20, dtype=torch.int64, device=device)
    confidence_counts = torch.zeros(5, dtype=torch.int64, device=device)
    confidence_edges = torch.tensor(
        [0.980, 0.990, 0.995, 0.999],
        dtype=torch.float32,
        device=device,
    )
    writer = EvaluationShardWriter(report_dir, rank) if report_dir is not None else None
    quick_local_limit = (
        (quick_error_limit + max(1, world_size) - 1) // max(1, world_size)
        if quick_error_limit > 0
        else 0
    )
    try:
        with torch.inference_mode():
            for batch in dataloader:
                images = batch["images"].to(device, non_blocking=True)
                if images.dtype == torch.uint8:
                    images = images.to(torch.float32).div_(255.0)
                labels = batch["labels"].to(device, non_blocking=True)
                logits = model(images[:, 0], images[:, 1])
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise ValueError(
                        f"Model must return [B,2], got {tuple(logits.shape)}"
                    )
                logits_fp32 = logits.float()
                batch_margins = logits_fp32[:, 1] - logits_fp32[:, 0]
                probabilities = torch.sigmoid(batch_margins)
                cross_entropy_sum += torch.nn.functional.cross_entropy(
                    logits_fp32, labels, reduction="sum"
                ).item()
                brier_sum += torch.square(
                    probabilities - labels.float()
                ).sum().item()
                sample_count += len(labels)
                predictions = batch_margins > cutoff
                local_counts[0] += int(((predictions) & (labels == 1)).sum().item())
                local_counts[1] += int(((predictions) & (labels == 0)).sum().item())
                local_counts[2] += int(((~predictions) & (labels == 1)).sum().item())
                local_counts[3] += int(((~predictions) & (labels == 0)).sum().item())

                auc_indices = torch.clamp(
                    (probabilities * auc_histogram_bins).long(),
                    max=auc_histogram_bins - 1,
                )
                positive_hist += torch.bincount(
                    auc_indices[labels == 1], minlength=auc_histogram_bins
                )
                negative_hist += torch.bincount(
                    auc_indices[labels == 0], minlength=auc_histogram_bins
                )
                calibration_indices = torch.clamp(
                    (probabilities * 20).long(), max=19
                )
                calibration_count += torch.bincount(
                    calibration_indices, minlength=20
                )
                calibration_probability.scatter_add_(
                    0, calibration_indices, probabilities
                )
                calibration_target.scatter_add_(
                    0, calibration_indices, labels.to(torch.int64)
                )
                confidence_indices = sum(
                    probabilities >= edge for edge in confidence_edges
                ).to(torch.int64)
                confidence_counts += torch.bincount(
                    confidence_indices, minlength=5
                )

                batch_metadata = [
                    _metadata_dict(item)
                    for item in batch.get("meta", [{}] * len(labels))
                ]
                batch_errors: list[dict] = []
                batch_near: list[dict] = []
                for logit, margin_tensor, target_tensor, probability_tensor, meta in zip(
                    logits_fp32.cpu(),
                    batch_margins.cpu(),
                    labels.cpu(),
                    probabilities.cpu(),
                    batch_metadata,
                ):
                    margin = float(margin_tensor)
                    target = int(target_tensor)
                    probability = float(probability_tensor)
                    prediction = int(margin > cutoff)
                    if exact_scores:
                        margins.append(margin)
                        targets.append(target)
                    counts = _counter(margin, target, cutoff)
                    game = str(meta.get("game", "unknown"))
                    video_id = str(meta.get("video_id", "unknown"))
                    _add_counter(group_game, game, counts)
                    _add_counter(group_video, (game, target, video_id), counts)
                    _add_counter(group_game_label, (game, target), counts)
                    _add_confidence(
                        group_video_confidence,
                        (game, target, video_id),
                        probability,
                    )
                    row = {
                        **meta,
                        "label": target,
                        "logit0": float(logit[0]),
                        "logit1": float(logit[1]),
                        "margin": margin,
                        "probability_class1": probability,
                        "prediction": prediction,
                        "checkpoint_step": checkpoint_step,
                    }
                    if prediction != target:
                        batch_errors.append(
                            {**row, "error_type": "FP" if prediction else "FN"}
                        )
                    band = _threshold_band(probability)
                    if band is not None:
                        batch_near.append({**row, "threshold_band": band})
                if evaluation_kind == "quick":
                    remaining_errors = max(
                        0, quick_local_limit - len(retained_errors)
                    )
                    remaining_near = max(
                        0, quick_local_limit - len(retained_near)
                    )
                    batch_errors = batch_errors[:remaining_errors]
                    batch_near = batch_near[:remaining_near]
                if writer is not None:
                    writer.write(batch_errors, batch_near)
                if writer is None or evaluation_kind == "quick":
                    retained_errors.extend(batch_errors)
                    retained_near.extend(batch_near)
    finally:
        if writer is not None:
            writer.close()

    if distributed:
        import torch.distributed as dist

        counts, floating = _make_reduction_tensors(
            local_counts,
            sample_count,
            cross_entropy_sum,
            brier_sum,
            device,
        )
        dist.all_reduce(counts)
        dist.all_reduce(floating)
        dist.all_reduce(positive_hist)
        dist.all_reduce(negative_hist)
        dist.all_reduce(calibration_count)
        dist.all_reduce(calibration_probability)
        dist.all_reduce(calibration_target)
        dist.all_reduce(confidence_counts)
        gathered_groups = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            (
                group_game,
                group_video,
                group_game_label,
                group_video_confidence,
            ),
            gathered_groups,
            dst=0,
        )
        gathered_scores = None
        if exact_scores:
            gathered_scores = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object((margins, targets), gathered_scores, dst=0)
        if rank != 0:
            return EvaluationOutput(None, None, retained_errors, retained_near)
        game_counters = _merge_group_counters(
            [payload[0] for payload in gathered_groups]
        )
        video_counters = _merge_group_counters(
            [payload[1] for payload in gathered_groups]
        )
        game_label_counters = _merge_group_counters(
            [payload[2] for payload in gathered_groups]
        )
        video_confidence = _merge_confidence(
            [payload[3] for payload in gathered_groups]
        )
        if exact_scores:
            margins = [
                margin for payload in gathered_scores for margin in payload[0]
            ]
            targets = [
                target for payload in gathered_scores for target in payload[1]
            ]
        tp, fp, fn, tn, sample_count = counts.tolist()
        cross_entropy_sum, brier_sum = floating.tolist()
    else:
        game_counters = group_game
        video_counters = group_video
        game_label_counters = group_game_label
        video_confidence = group_video_confidence
        tp, fp, fn, tn = local_counts

    if exact_scores:
        ranking = confusion_from_margins(margins, targets, threshold)
        roc_auc, pr_auc = ranking.roc_auc, ranking.pr_auc
        auc_method = "exact"
    else:
        roc_auc, pr_auc = _histogram_auc(
            positive_hist.cpu().tolist(), negative_hist.cpu().tolist()
        )
        auc_method = f"histogram_{auc_histogram_bins}_bins"
    global_binary = metrics_from_counts(
        int(tp),
        int(fp),
        int(fn),
        int(tn),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )
    by_game = _group_rows(game_counters, ("game",))
    by_video = _video_rows(video_counters, video_confidence)
    by_game_label = _group_rows(game_label_counters, ("game", "label"))
    calibration_counts = calibration_count.cpu().tolist()
    calibration_probabilities = calibration_probability.cpu().tolist()
    calibration_targets = calibration_target.cpu().tolist()
    metrics = global_binary.to_dict()
    metrics.update(
        {
            "cross_entropy": cross_entropy_sum / sample_count if sample_count else 0.0,
            "brier_score": brier_sum / sample_count if sample_count else 0.0,
            "ece_20_bins": _calibration_metrics(
                calibration_counts,
                calibration_probabilities,
                calibration_targets,
                int(sample_count),
            ),
            "auc_method": auc_method,
            "sample_count": int(sample_count),
            "threshold": threshold,
            "threshold_is_business_score": True,
            "global_f1_tau099": global_binary.f1,
            "macro_game_f1_tau099": (
                sum(row["f1"] for row in by_game) / len(by_game) if by_game else 0.0
            ),
            "worst_game_f1_tau099": (
                min(row["f1"] for row in by_game) if by_game else 0.0
            ),
            "confidence_histogram": dict(
                zip(
                    ["<0.980", "0.980-0.990", "0.990-0.995", "0.995-0.999", ">=0.999"],
                    confidence_counts.cpu().tolist(),
                )
            ),
        }
    )
    return EvaluationOutput(
        metrics,
        {
            "by_game": by_game,
            "by_video": by_video,
            "by_game_label": by_game_label,
        },
        retained_errors,
        retained_near,
    )
