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
    if 0.990 < probability < 0.995:
        return "0.990-0.995"
    if 0.995 <= probability < 0.999:
        return "0.995-0.999"
    if probability >= 0.999:
        return ">=0.999"
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
) -> EvaluationOutput:
    import torch

    model.eval()
    cutoff = probability_threshold_to_margin(threshold)
    margins: list[float] = []
    targets: list[int] = []
    errors: list[dict] = []
    near_threshold: list[dict] = []
    group_game: dict = {}
    group_video: dict = {}
    group_game_label: dict = {}
    cross_entropy_sum = 0.0
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            if images.dtype == torch.uint8:
                images = images.to(torch.float32).div_(255.0)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(images[:, 0], images[:, 1])
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Model must return [B,2], got {tuple(logits.shape)}")
            logits_fp32 = logits.float()
            batch_margins = logits_fp32[:, 1] - logits_fp32[:, 0]
            cross_entropy_sum += torch.nn.functional.cross_entropy(
                logits_fp32, labels, reduction="sum"
            ).item()
            batch_metadata = [
                _metadata_dict(item)
                for item in batch.get("meta", [{}] * len(batch_margins))
            ]
            for logit, margin_tensor, target_tensor, meta in zip(
                logits_fp32.cpu(),
                batch_margins.cpu(),
                labels.cpu(),
                batch_metadata,
            ):
                margin = float(margin_tensor)
                target = int(target_tensor)
                prediction = int(margin > cutoff)
                probability = probability_from_margin(margin)
                margins.append(margin)
                targets.append(target)
                counts = _counter(margin, target, cutoff)
                game = str(meta.get("game", "unknown"))
                video_id = str(meta.get("video_id", "unknown"))
                _add_counter(group_game, game, counts)
                _add_counter(group_video, (game, video_id), counts)
                _add_counter(group_game_label, (game, target), counts)
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
                    errors.append(
                        {**row, "error_type": "FP" if prediction else "FN"}
                    )
                band = _threshold_band(probability)
                if band is not None:
                    near_threshold.append({**row, "threshold_band": band})

    local_metrics = confusion_from_margins(margins, targets, threshold)
    global_margins = margins
    global_targets = targets
    game_counters = group_game
    video_counters = group_video
    game_label_counters = group_game_label
    sample_count = len(targets)
    if distributed:
        import torch.distributed as dist

        numeric = torch.tensor(
            [
                local_metrics.tp,
                local_metrics.fp,
                local_metrics.fn,
                local_metrics.tn,
                cross_entropy_sum,
                sample_count,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(numeric)
        gathered_scores = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object((margins, targets), gathered_scores, dst=0)
        gathered_groups = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            (group_game, group_video, group_game_label), gathered_groups, dst=0
        )
        if rank != 0:
            return EvaluationOutput(None, None, errors, near_threshold)
        global_margins = [
            margin for payload in gathered_scores for margin in payload[0]
        ]
        global_targets = [
            target for payload in gathered_scores for target in payload[1]
        ]
        game_counters = _merge_group_counters(
            [payload[0] for payload in gathered_groups]
        )
        video_counters = _merge_group_counters(
            [payload[1] for payload in gathered_groups]
        )
        game_label_counters = _merge_group_counters(
            [payload[2] for payload in gathered_groups]
        )
        tp, fp, fn, tn, cross_entropy_sum, sample_count = numeric.tolist()
        ranking = confusion_from_margins(global_margins, global_targets, threshold)
        global_binary = metrics_from_counts(
            int(tp),
            int(fp),
            int(fn),
            int(tn),
            roc_auc=ranking.roc_auc,
            pr_auc=ranking.pr_auc,
        )
    else:
        global_binary = local_metrics

    by_game = _group_rows(game_counters, ("game",))
    by_video = _group_rows(video_counters, ("game", "video_id"))
    by_game_label = _group_rows(game_label_counters, ("game", "label"))
    metrics = global_binary.to_dict()
    metrics.update(
        {
            "cross_entropy": cross_entropy_sum / sample_count if sample_count else 0.0,
            "sample_count": int(sample_count),
            "threshold": threshold,
            "global_f1_tau099": global_binary.f1,
            "macro_game_f1_tau099": (
                sum(row["f1"] for row in by_game) / len(by_game) if by_game else 0.0
            ),
            "worst_game_f1_tau099": (
                min(row["f1"] for row in by_game) if by_game else 0.0
            ),
            "macro_video_f1_tau099": (
                sum(row["f1"] for row in by_video) / len(by_video)
                if by_video
                else 0.0
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
        errors,
        near_threshold,
    )
