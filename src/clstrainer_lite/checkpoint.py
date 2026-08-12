from __future__ import annotations

import os
from pathlib import Path

import torch

from .distributed import unwrap_model


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def save_model_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Save a pure state_dict, safe and easy to reuse outside this trainer."""
    _atomic_torch_save(unwrap_model(model).state_dict(), Path(path))


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epoch: int,
    global_step: int,
    history: list[dict],
    config: dict,
    path: str | Path,
) -> None:
    _atomic_torch_save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "history": history,
            "config": config,
        },
        Path(path),
    )
