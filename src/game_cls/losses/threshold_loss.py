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


def threshold_weight_from_steps(
    step: int,
    warmup_steps: int,
    ramp_steps: int,
    max_weight: float,
) -> float:
    """Explicit-step schedule, independent of the total step budget.

    Unlike the ratio schedule, extending ``max_steps`` does not move the
    absolute step at which the margin loss activates.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    warmup_steps = max(0, int(warmup_steps))
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return max_weight
    progressed = step - warmup_steps
    if progressed >= ramp_steps:
        return max_weight
    return max_weight * progressed / ramp_steps


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


def negative_tail_ohem_loss(
    logits,
    target,
    *,
    threshold: float = 0.99,
    safety_margin: float = 0.20,
    temperature: float = 0.50,
    hard_negative_k: int | None = None,
):
    """Per-sample negative margin loss, optionally focused on hardest negatives.

    Only negative samples (``target < 0.5``) contribute. When ``hard_negative_k``
    is set and there are more negatives than ``hard_negative_k``, the mean is
    taken over the hardest ``hard_negative_k`` negatives instead of all of them.
    """
    import torch
    from torch.nn import functional as F

    if tuple(logits.shape[-1:]) != (2,):
        raise ValueError(f"Expected logits shaped [B,2], got {tuple(logits.shape)}")
    threshold_margin = probability_threshold_to_margin(threshold)
    margin = logits[:, 1].float() - logits[:, 0].float()
    target = target.float()
    negative_loss = temperature * F.softplus(
        (margin - threshold_margin + safety_margin) / temperature
    )
    negatives = negative_loss[target < 0.5]
    if negatives.numel() == 0:
        return logits.new_zeros(())
    if hard_negative_k and negatives.numel() > hard_negative_k:
        return torch.topk(negatives, hard_negative_k, dim=0).values.mean()
    return negatives.mean()


def pairwise_ranking_loss(
    logits,
    target,
    *,
    rank_margin: float = 0.2,
    threshold: float = 0.99,
):
    """Mean max-margin penalty: each positive must beat hard negatives.

    A hard negative is a negative whose margin is at or above the decision
    boundary (``probability_threshold_to_margin(threshold)``) minus
    ``rank_margin``. For each positive the penalty is the largest
    ``relu(rank_margin - positive_margin + hard_negative_margin)`` over hard
    negatives; the result is the mean over positives.
    """
    if tuple(logits.shape[-1:]) != (2,):
        raise ValueError(f"Expected logits shaped [B,2], got {tuple(logits.shape)}")
    margin = logits[:, 1].float() - logits[:, 0].float()
    target = target.float()
    positives = margin[target > 0.5]
    negatives = margin[target < 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.new_zeros(())
    threshold_margin = probability_threshold_to_margin(threshold)
    hard_negatives = negatives[negatives >= threshold_margin - rank_margin]
    if hard_negatives.numel() == 0:
        return logits.new_zeros(())
    penalty = (
        rank_margin - positives.unsqueeze(1) + hard_negatives.unsqueeze(0)
    ).clamp(min=0.0)
    return penalty.max(dim=1).values.mean()


def combined_loss(logits, target, config: dict, step: int, total_steps: int):
    from torch.nn import functional as F

    label_smoothing = float(config.get("label_smoothing", 0.0))
    if label_smoothing:
        ce = F.cross_entropy(
            logits.float(), target, label_smoothing=label_smoothing
        )
    else:
        ce = F.cross_entropy(logits.float(), target)
    threshold_component = threshold_margin_loss(
        logits,
        target,
        threshold=config.get("threshold", 0.99),
        safety_margin=config.get("threshold_safety_margin", 0.20),
        temperature=config.get("threshold_temperature", 0.50),
    )
    max_weight = config.get("threshold_loss_weight", 0.20)
    warmup_steps = config.get("threshold_warmup_steps")
    ramp_steps = config.get("threshold_ramp_steps")
    if warmup_steps is not None or ramp_steps is not None:
        threshold_weight = threshold_weight_from_steps(
            step,
            int(warmup_steps or 0),
            int(ramp_steps or 0),
            max_weight,
        )
    else:
        threshold_weight = threshold_weight_at_step(
            step,
            total_steps,
            max_weight,
            config.get("threshold_warmup_ratio", 0.10),
            config.get("threshold_ramp_ratio", 0.20),
        )
    negative_tail_weight = float(config.get("negative_tail_loss_weight", 0.0))
    negative_tail_k = config.get("negative_tail_hard_negative_k")
    rank_weight = float(config.get("rank_loss_weight", 0.0))
    rank_margin = float(config.get("rank_margin", 0.2))
    negative_tail_component = None
    rank_component = None
    if negative_tail_weight > 0:
        negative_tail_component = negative_tail_ohem_loss(
            logits,
            target,
            threshold=config.get("threshold", 0.99),
            safety_margin=config.get("threshold_safety_margin", 0.20),
            temperature=config.get("threshold_temperature", 0.50),
            hard_negative_k=negative_tail_k,
        )
    if rank_weight > 0:
        rank_component = pairwise_ranking_loss(
            logits,
            target,
            rank_margin=rank_margin,
            threshold=config.get("threshold", 0.99),
        )
    total = config.get("cross_entropy_weight", 1.0) * ce
    total = total + threshold_weight * threshold_component
    if negative_tail_component is not None:
        total = total + negative_tail_weight * negative_tail_component
    if rank_component is not None:
        total = total + rank_weight * rank_component
    return total, {
        "cross_entropy": ce.detach(),
        "threshold_loss": threshold_component.detach(),
        "threshold_weight": threshold_weight,
        "negative_tail_loss": (
            negative_tail_component.detach()
            if negative_tail_component is not None
            else logits.new_zeros(())
        ),
        "rank_loss": (
            rank_component.detach()
            if rank_component is not None
            else logits.new_zeros(())
        ),
    }

