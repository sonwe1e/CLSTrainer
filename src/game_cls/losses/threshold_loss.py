from __future__ import annotations

import math


def probability_threshold_to_margin(threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    return math.log(threshold / (1.0 - threshold))


def threshold_weight_at_step(
    step: int,
    total_steps: int,
    max_weight: float,
    warmup_ratio: float,
    ramp_ratio: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = max(0.0, min(1.0, step / total_steps))
    if progress <= warmup_ratio:
        return 0.0
    if ramp_ratio <= 0 or progress >= warmup_ratio + ramp_ratio:
        return max_weight
    return max_weight * (progress - warmup_ratio) / ramp_ratio


def threshold_margin_loss(
    logits,
    target,
    threshold: float = 0.99,
    safety_margin: float = 0.20,
    temperature: float = 0.50,
):
    import torch
    from torch.nn import functional as F

    if tuple(logits.shape[-1:]) != (2,):
        raise ValueError(f"Expected logits shaped [B,2], got {tuple(logits.shape)}")
    threshold_margin = probability_threshold_to_margin(threshold)
    margin = logits[:, 1].float() - logits[:, 0].float()
    target = target.float()
    positive_loss = temperature * F.softplus(
        (threshold_margin + safety_margin - margin) / temperature
    )
    negative_loss = temperature * F.softplus(
        (margin - threshold_margin + safety_margin) / temperature
    )
    return torch.where(target > 0.5, positive_loss, negative_loss).mean()


def combined_loss(logits, target, config: dict, step: int, total_steps: int):
    from torch.nn import functional as F

    ce = F.cross_entropy(logits.float(), target)
    threshold_component = threshold_margin_loss(
        logits,
        target,
        threshold=config.get("threshold", 0.99),
        safety_margin=config.get("threshold_safety_margin", 0.20),
        temperature=config.get("threshold_temperature", 0.50),
    )
    threshold_weight = threshold_weight_at_step(
        step,
        total_steps,
        config.get("threshold_loss_weight", 0.20),
        config.get("threshold_warmup_ratio", 0.10),
        config.get("threshold_ramp_ratio", 0.20),
    )
    total = config.get("cross_entropy_weight", 1.0) * ce
    total = total + threshold_weight * threshold_component
    return total, {
        "cross_entropy": ce.detach(),
        "threshold_loss": threshold_component.detach(),
        "threshold_weight": threshold_weight,
    }

