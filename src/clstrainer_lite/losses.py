"""Drop-in multi-class Focal Loss for CLSTrainer-Lite.

Expected logits: [B, C]
Expected targets: [B] int64 class ids

For binary CLSTrainer-Lite use C=2 and alpha=[class0_weight, class1_weight].
The alpha values are class weights; they do not need to sum to 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float = 1.5,
        alpha: Sequence[float] | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if tensor.ndim != 1 or tensor.numel() < 2:
                raise ValueError("alpha must be a 1-D class-weight vector")
            if not torch.isfinite(tensor).all() or (tensor <= 0).any():
                raise ValueError("all alpha values must be finite and > 0")
            self.register_buffer("alpha", tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"targets must be [B] matching logits, got {tuple(targets.shape)}"
            )

        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # -log(pt), multiplied by the focal modulation term.
        loss = -(1.0 - pt).pow(self.gamma) * log_pt

        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError(
                    f"alpha has {self.alpha.numel()} entries but logits have "
                    f"{logits.shape[1]} classes"
                )
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            loss = loss * alpha.gather(0, targets)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.alpha is not None:
            # Match torch.nn.CrossEntropyLoss(weight=...) at gamma=0: the
            # weighted mean is normalized by the sum of target class weights.
            normalizer = alpha.gather(0, targets).sum().clamp_min(1e-12)
            return loss.sum() / normalizer
        return loss.mean()
