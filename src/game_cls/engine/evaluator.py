from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from game_cls.engine.device import autocast_context
from game_cls.losses.threshold_loss import (
    probability_threshold_to_margin,
    threshold_margin_loss,
)
from game_cls.metrics.binary_metrics import (
    confusion_from_margins,
    metrics_from_counts,
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
        rows.append({**dict(zip(key_names, key, strict=False)), **metrics})
    return rows


def _game_label_rows(counters: dict) -> list[dict]:
    rows = []
    for (game, label), counts in sorted(counters.items()):
        metrics = metrics_from_counts(*counts).to_dict()
        row = {
            "game": game,
            "label": label,
            "sample_count": sum(counts),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }
        if label == 1:
            row.update(
                {
                    "primary_metric": "positive_recall",
                    "positive_recall": metrics["recall"],
                    "fn_rate": 1.0 - metrics["recall"],
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
                    "negative_specificity": metrics["specificity"],
                    "fp_rate": 1.0 - metrics["specificity"],
                }
            )
        rows.append(row)
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
        for positive, negative in zip(positive_histogram, negative_histogram, strict=False):
            concordant += positive * (negatives_below + 0.5 * negative)
            negatives_below += negative
        roc_auc = concordant / (positives * negatives)
    pr_auc = 0.0
    if positives:
        tp = fp = 0
        previous_recall = 0.0
        for positive, negative in zip(
            reversed(positive_histogram), reversed(negative_histogram), strict=False
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
        counts, probability_sums, target_sums, strict=False
    ):
        if count:
            confidence = probability_sum / count
            accuracy = target_sum / count
            ece += count / sample_count * abs(confidence - accuracy)
    return ece


def _make_reduction_tensors(
    local_counts,
    sample_count,
    cross_entropy_sum,
    brier_sum,
    threshold_loss_sum,
    device,
):
    import torch

    counts = torch.cat(
        (
            torch.as_tensor(local_counts, dtype=torch.int64, device=device),
            torch.tensor([sample_count], dtype=torch.int64, device=device),
        )
    )
    floating = torch.stack(
        (
            torch.as_tensor(
                cross_entropy_sum, dtype=torch.float32, device=device
            ),
            torch.as_tensor(brier_sum, dtype=torch.float32, device=device),
            torch.as_tensor(
                threshold_loss_sum, dtype=torch.float32, device=device
            ),
        )
    )
    return counts, floating


def _histogram_margin_percentiles(
    histogram,
    bins: int,
    quantiles: tuple[float, ...],
) -> list[float | None]:
    """Margin-space percentiles from a probability histogram.

    The histogram counts class probabilities in ``bins`` uniform bins; each
    bin center is mapped back to margin space via log(p / (1 - p)).
    """
    import numpy as np

    counts = np.asarray(histogram, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        return [None] * len(quantiles)
    cdf = np.cumsum(counts)
    results: list[float | None] = []
    for quantile in quantiles:
        target = max(1, int(math.ceil(quantile * total)))
        bin_index = int(np.searchsorted(cdf, target, side="left"))
        bin_index = min(max(bin_index, 0), bins - 1)
        probability = min(max((bin_index + 0.5) / bins, 1e-6), 1.0 - 1e-6)
        results.append(math.log(probability / (1.0 - probability)))
    return results


def _build_group_dicts(
    group_catalogs: dict,
    game_counts_array,
    game_label_counts_array,
    video_counts_array,
    video_probability_count,
    video_probability_sum,
    video_probability_min,
    video_probability_max,
) -> tuple[dict, dict, dict, dict]:
    """Turn per-catalog index arrays into the grouped-metrics dicts.

    All ranks share the same catalog (the dataset provides it), so the
    arrays are reduced as tensors first; dicts are only materialized on
    the rank that reports.
    """
    group_game = {
        key: values.tolist()
        for key, values in zip(
            group_catalogs["game"], game_counts_array, strict=False
        )
        if values.sum()
    }
    group_game_label = {
        tuple(key): values.tolist()
        for key, values in zip(
            group_catalogs["game_label"],
            game_label_counts_array,
            strict=False,
        )
        if values.sum()
    }
    group_video = {
        tuple(key): values.tolist()
        for key, values in zip(
            group_catalogs["video"], video_counts_array, strict=False
        )
        if values.sum()
    }
    group_video_confidence = {
        tuple(key): {
            "count": int(video_probability_count[index]),
            "sum": float(video_probability_sum[index]),
            "min": float(video_probability_min[index]),
            "max": float(video_probability_max[index]),
        }
        for index, key in enumerate(group_catalogs["video"])
        if video_probability_count[index]
    }
    return group_game, group_video, group_game_label, group_video_confidence


def _all_reduce_group_arrays(
    dist, sum_arrays, min_array, max_array
) -> None:
    """Tensor-reduce the per-catalog numpy arrays in place.

    Counts/sums use addition; the per-video probability min/max use the
    MIN/MAX reduction ops (arrays are initialized to +/-inf).
    """
    import numpy as np
    import torch

    for array in sum_arrays:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        dist.all_reduce(tensor)
        array[...] = tensor.numpy()
    min_tensor = torch.from_numpy(np.ascontiguousarray(min_array))
    dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
    min_array[...] = min_tensor.numpy()
    max_tensor = torch.from_numpy(np.ascontiguousarray(max_array))
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
    max_array[...] = max_tensor.numpy()


# Exact-AUC distributed evaluation gathers raw score lists to rank 0; cap
# the sample count so a huge test set cannot turn into a communication and
# rank-0 memory bottleneck. Histogram AUC is the default for large sets.
EXACT_AUC_MAX_SAMPLES = 2_000_000




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
    amp: bool = False,
    amp_dtype: str = "bfloat16",
    parquet_row_group_size: int = 4096,
    group_catalogs: dict | None = None,
    threshold_loss_weight: float = 0.0,
    cross_entropy_weight: float = 1.0,
    threshold_safety_margin: float = 0.20,
    threshold_temperature: float = 0.50,
) -> EvaluationOutput:
    import numpy as np
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
    vectorized_groups = bool(group_catalogs)
    if vectorized_groups:
        game_counts_array = np.zeros(
            (len(group_catalogs["game"]), 4), dtype=np.int64
        )
        game_label_counts_array = np.zeros(
            (len(group_catalogs["game_label"]), 4), dtype=np.int64
        )
        video_counts_array = np.zeros(
            (len(group_catalogs["video"]), 4), dtype=np.int64
        )
        video_probability_count = np.zeros(
            len(group_catalogs["video"]), dtype=np.int64
        )
        video_probability_sum = np.zeros(
            len(group_catalogs["video"]), dtype=np.float64
        )
        video_probability_min = np.full(
            len(group_catalogs["video"]), np.inf, dtype=np.float64
        )
        video_probability_max = np.full(
            len(group_catalogs["video"]), -np.inf, dtype=np.float64
        )
    cross_entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
    brier_sum = torch.zeros((), dtype=torch.float32, device=device)
    threshold_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = 0
    local_counts = torch.zeros(4, dtype=torch.int64, device=device)
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
    writer = (
        EvaluationShardWriter(
            report_dir,
            rank,
            row_group_size=parquet_row_group_size,
        )
        if report_dir is not None
        else None
    )
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
                    compute_dtype = (
                        torch.bfloat16
                        if amp and amp_dtype == "bfloat16"
                        else torch.float16
                        if amp and amp_dtype == "float16"
                        else torch.float32
                    )
                    images = images.to(compute_dtype).div_(255.0)
                labels = batch["labels"].to(device, non_blocking=True)
                with autocast_context(device, amp, amp_dtype):
                    logits = model(images[:, 0], images[:, 1])
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise ValueError(
                        f"Model must return [B,2], got {tuple(logits.shape)}"
                    )
                logits_fp32 = logits.float()
                batch_margins = logits_fp32[:, 1] - logits_fp32[:, 0]
                probabilities = torch.sigmoid(batch_margins)
                cross_entropy_sum.add_(torch.nn.functional.cross_entropy(
                    logits_fp32, labels, reduction="sum"
                ))
                brier_sum.add_(torch.square(
                    probabilities - labels.float()
                ).sum())
                threshold_loss_sum.add_(
                    threshold_margin_loss(
                        logits_fp32,
                        labels,
                        threshold=threshold,
                        safety_margin=threshold_safety_margin,
                        temperature=threshold_temperature,
                    )
                    * len(labels)
                )
                sample_count += len(labels)
                predictions = batch_margins > cutoff
                local_counts.add_(
                    torch.stack(
                        (
                            (predictions & (labels == 1)).sum(),
                            (predictions & (labels == 0)).sum(),
                            ((~predictions) & (labels == 1)).sum(),
                            ((~predictions) & (labels == 0)).sum(),
                        )
                    ).to(torch.int64)
                )

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

                raw_metadata = batch.get("meta", [{}] * len(labels))
                batch_metadata = (
                    raw_metadata
                    if vectorized_groups
                    else [_metadata_dict(item) for item in raw_metadata]
                )
                batch_errors: list[dict] = []
                batch_near: list[dict] = []
                compact = torch.stack(
                    (
                        batch_margins,
                        probabilities,
                        labels.to(torch.float32),
                        predictions.to(torch.float32),
                    ),
                    dim=1,
                ).cpu()
                compact_numpy = compact.numpy()
                targets_numpy = compact_numpy[:, 2].astype(
                    np.int64, copy=False
                )
                predictions_numpy = compact_numpy[:, 3].astype(
                    np.int64, copy=False
                )
                if exact_scores:
                    margins.extend(compact_numpy[:, 0].tolist())
                    targets.extend(targets_numpy.tolist())
                if vectorized_groups:
                    required_group_fields = (
                        "game_id",
                        "game_label_id",
                        "video_group_id",
                    )
                    missing_group_fields = [
                        key
                        for key in required_group_fields
                        if key not in batch
                    ]
                    if missing_group_fields:
                        raise KeyError(
                            "Vectorized evaluation is missing group IDs: "
                            f"{missing_group_fields}"
                        )
                    outcomes = np.empty(len(targets_numpy), dtype=np.int64)
                    outcomes[
                        (predictions_numpy == 1) & (targets_numpy == 1)
                    ] = 0
                    outcomes[
                        (predictions_numpy == 1) & (targets_numpy == 0)
                    ] = 1
                    outcomes[
                        (predictions_numpy == 0) & (targets_numpy == 1)
                    ] = 2
                    outcomes[
                        (predictions_numpy == 0) & (targets_numpy == 0)
                    ] = 3

                    def accumulate_counts(
                        destination, ids, outcomes=outcomes
                    ) -> None:
                        ids = ids.numpy().astype(np.int64, copy=False)
                        flattened = np.bincount(
                            ids * 4 + outcomes,
                            minlength=destination.size,
                        )
                        if flattened.size != destination.size:
                            raise IndexError(
                                "Evaluation group ID exceeds its catalog"
                            )
                        destination += flattened.reshape(
                            destination.shape
                        )

                    accumulate_counts(
                        game_counts_array, batch["game_id"]
                    )
                    accumulate_counts(
                        game_label_counts_array,
                        batch["game_label_id"],
                    )
                    accumulate_counts(
                        video_counts_array, batch["video_group_id"]
                    )
                    video_ids = (
                        batch["video_group_id"]
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    probability_values = compact_numpy[:, 1].astype(
                        np.float64, copy=False
                    )
                    np.add.at(video_probability_count, video_ids, 1)
                    np.add.at(
                        video_probability_sum,
                        video_ids,
                        probability_values,
                    )
                    np.minimum.at(
                        video_probability_min,
                        video_ids,
                        probability_values,
                    )
                    np.maximum.at(
                        video_probability_max,
                        video_ids,
                        probability_values,
                    )
                else:
                    for values, meta in zip(
                        compact_numpy, batch_metadata, strict=False
                    ):
                        margin = float(values[0])
                        probability = float(values[1])
                        target = int(values[2])
                        counts = _counter(margin, target, cutoff)
                        game = str(meta.get("game", "unknown"))
                        video_id = str(
                            meta.get("video_id", "unknown")
                        )
                        _add_counter(group_game, game, counts)
                        _add_counter(
                            group_video,
                            (game, target, video_id),
                            counts,
                        )
                        _add_counter(
                            group_game_label, (game, target), counts
                        )
                        _add_confidence(
                            group_video_confidence,
                            (game, target, video_id),
                            probability,
                        )
                report_mask = (
                    (predictions != labels.bool())
                    | ((probabilities >= 0.980) & (probabilities <= 0.995))
                )
                report_indices = report_mask.nonzero(as_tuple=False).flatten()
                report_logits = logits_fp32.index_select(
                    0, report_indices
                ).cpu()
                for index, logit in zip(
                    report_indices.cpu().tolist(), report_logits, strict=False
                ):
                    values = compact_numpy[index]
                    meta = _metadata_dict(batch_metadata[index])
                    margin = float(values[0])
                    probability = float(values[1])
                    target = int(values[2])
                    prediction = int(values[3])
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

    if vectorized_groups and not distributed:
        group_game, group_video, group_game_label, group_video_confidence = (
            _build_group_dicts(
                group_catalogs,
                game_counts_array,
                game_label_counts_array,
                video_counts_array,
                video_probability_count,
                video_probability_sum,
                video_probability_min,
                video_probability_max,
            )
        )
    if distributed:
        import torch.distributed as dist

        counts, floating = _make_reduction_tensors(
            local_counts,
            sample_count,
            cross_entropy_sum,
            brier_sum,
            threshold_loss_sum,
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
        if vectorized_groups:
            # All ranks share the same catalog, so the per-catalog index
            # arrays reduce directly as tensors instead of gathering large
            # Python dicts to rank 0.
            _all_reduce_group_arrays(
                dist,
                (
                    game_counts_array,
                    game_label_counts_array,
                    video_counts_array,
                    video_probability_count,
                    video_probability_sum,
                ),
                video_probability_min,
                video_probability_max,
            )
            group_game, group_video, group_game_label, group_video_confidence = (
                _build_group_dicts(
                    group_catalogs,
                    game_counts_array,
                    game_label_counts_array,
                    video_counts_array,
                    video_probability_count,
                    video_probability_sum,
                    video_probability_min,
                    video_probability_max,
                )
            )
        gathered_scores = None
        if exact_scores:
            if full_auc_mode == "exact" and int(sample_count) > EXACT_AUC_MAX_SAMPLES:
                raise ValueError(
                    "Distributed exact-AUC evaluation gathers all scores to "
                    f"rank 0: {int(sample_count)} samples exceeds the "
                    f"{EXACT_AUC_MAX_SAMPLES} cap. Use "
                    "evaluation.full_auc_mode=histogram for large test sets."
                )
            gathered_scores = (
                [None for _ in range(world_size)] if rank == 0 else None
            )
            dist.gather_object((margins, targets), gathered_scores, dst=0)
        if not vectorized_groups:
            # No shared catalog: the per-rank dicts are small and are still
            # merged through gather_object (every rank participates).
            gathered_groups = (
                [None for _ in range(world_size)] if rank == 0 else None
            )
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
        if rank != 0:
            return EvaluationOutput(None, None, retained_errors, retained_near)
        if vectorized_groups:
            game_counters = group_game
            video_counters = group_video
            game_label_counters = group_game_label
            video_confidence = group_video_confidence
        else:
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
        (
            cross_entropy_sum,
            brier_sum,
            threshold_loss_sum,
        ) = floating.tolist()
        positive_histogram = positive_hist.cpu().tolist()
        negative_histogram = negative_hist.cpu().tolist()
    else:
        game_counters = group_game
        video_counters = group_video
        game_label_counters = group_game_label
        video_confidence = group_video_confidence
        tp, fp, fn, tn = local_counts.tolist()
        (
            cross_entropy_sum,
            brier_sum,
            threshold_loss_sum,
        ) = torch.stack(
            (cross_entropy_sum, brier_sum, threshold_loss_sum)
        ).tolist()
        positive_histogram = positive_hist.cpu().tolist()
        negative_histogram = negative_hist.cpu().tolist()

    if exact_scores:
        ranking = confusion_from_margins(margins, targets, threshold)
        roc_auc, pr_auc = ranking.roc_auc, ranking.pr_auc
        auc_method = "exact"
    else:
        roc_auc, pr_auc = _histogram_auc(
            positive_histogram, negative_histogram
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
    by_game_label = _game_label_rows(game_label_counters)
    calibration_counts = calibration_count.cpu().tolist()
    calibration_probabilities = calibration_probability.cpu().tolist()
    calibration_targets = calibration_target.cpu().tolist()
    metrics = global_binary.to_dict()
    cross_entropy_mean = (
        cross_entropy_sum / sample_count if sample_count else 0.0
    )
    threshold_loss_mean = (
        threshold_loss_sum / sample_count if sample_count else 0.0
    )
    positive_total = int(tp) + int(fn)
    negative_total = int(fp) + int(tn)
    positive_percentiles = _histogram_margin_percentiles(
        positive_histogram, auc_histogram_bins, (0.10, 0.50, 0.90)
    )
    negative_percentiles = _histogram_margin_percentiles(
        negative_histogram, auc_histogram_bins, (0.10, 0.50, 0.90)
    )
    metrics.update(
        {
            "cross_entropy": cross_entropy_mean,
            "threshold_loss": threshold_loss_mean,
            "threshold_loss_weight": float(threshold_loss_weight),
            "objective_loss": float(cross_entropy_weight) * cross_entropy_mean
            + float(threshold_loss_weight) * threshold_loss_mean,
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
            # Neutral names: the decision threshold is configurable, so the
            # metric names must not bake in a fixed 0.99. Legacy _tau099
            # aliases are kept for backward compatibility with old reports
            # and configs.
            "global_f1_at_decision_threshold": global_binary.f1,
            "macro_game_f1_at_decision_threshold": (
                sum(row["f1"] for row in by_game) / len(by_game) if by_game else 0.0
            ),
            "worst_game_f1_at_decision_threshold": (
                min(row["f1"] for row in by_game) if by_game else 0.0
            ),
            "global_f1_tau099": global_binary.f1,
            "macro_game_f1_tau099": (
                sum(row["f1"] for row in by_game) / len(by_game) if by_game else 0.0
            ),
            "worst_game_f1_tau099": (
                min(row["f1"] for row in by_game) if by_game else 0.0
            ),
            "positive_margin_pass_rate": (
                int(tp) / positive_total if positive_total else None
            ),
            "negative_margin_pass_rate": (
                int(tn) / negative_total if negative_total else None
            ),
            "positive_margin_p10": positive_percentiles[0],
            "positive_margin_p50": positive_percentiles[1],
            "positive_margin_p90": positive_percentiles[2],
            "negative_margin_p10": negative_percentiles[0],
            "negative_margin_p50": negative_percentiles[1],
            "negative_margin_p90": negative_percentiles[2],
            "confidence_histogram": dict(
                zip(
                    ["<0.980", "0.980-0.990", "0.990-0.995", "0.995-0.999", ">=0.999"],
                    confidence_counts.cpu().tolist(), strict=False,
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
