from __future__ import annotations

from dataclasses import asdict, is_dataclass

from game_cls.metrics.binary_metrics import (
    confusion_from_margins,
    probability_from_margin,
)


def evaluate(model, dataloader, device, threshold: float = 0.99):
    import torch

    model.eval()
    margins: list[float] = []
    targets: list[int] = []
    metadata: list[dict] = []
    errors: list[dict] = []
    cross_entropy_sum = 0.0
    with torch.inference_mode():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            images = images.float().div_(255.0) if images.dtype == torch.uint8 else images
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(images[:, 0], images[:, 1])
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Model must return [B,2], got {tuple(logits.shape)}")
            batch_margins = (logits[:, 1].float() - logits[:, 0].float()).cpu()
            cross_entropy_sum += torch.nn.functional.cross_entropy(
                logits.float(), labels, reduction="sum"
            ).item()
            margins.extend(batch_margins.tolist())
            targets.extend(labels.cpu().tolist())
            for item in batch.get("meta", [{}] * len(batch_margins)):
                if is_dataclass(item):
                    metadata.append(asdict(item))
                elif isinstance(item, dict):
                    metadata.append(dict(item))
                else:
                    metadata.append({"sample": str(item)})
    metrics = confusion_from_margins(margins, targets, threshold)
    metrics_dict = asdict(metrics)
    metrics_dict["cross_entropy"] = (
        cross_entropy_sum / len(targets) if targets else 0.0
    )
    cutoff = __import__(
        "game_cls.losses.threshold_loss", fromlist=["probability_threshold_to_margin"]
    ).probability_threshold_to_margin(threshold)
    if not metadata:
        metadata = [{} for _ in margins]
    for margin, target, meta in zip(margins, targets, metadata):
        prediction = int(margin > cutoff)
        if prediction != target:
            errors.append({**meta,
                "label": target,
                "prediction": prediction,
                "margin": margin,
                "probability_class1": probability_from_margin(margin),
                "error_type": "FP" if prediction else "FN",
            })
    return metrics_dict, errors
