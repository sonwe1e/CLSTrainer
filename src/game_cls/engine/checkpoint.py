from __future__ import annotations

import os
from pathlib import Path
import random


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _atomic_torch_save(payload, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    # Windows requires a writable file descriptor for fsync.
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint_pair(
    output_dir: str | Path,
    tag: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_metrics: dict,
    config: dict,
) -> None:
    import numpy as np
    import torch

    output_dir = Path(output_dir)
    state_dict = unwrap_model(model).state_dict()
    _atomic_torch_save(state_dict, output_dir / f"model_{tag}.pth")
    _atomic_torch_save(
        {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "global_step": global_step,
            "sampler_epoch": epoch,
            "best_metrics": best_metrics,
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "config": config,
        },
        output_dir / f"checkpoint_{tag}.pth",
    )


def restore_training_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, scaler=None
) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
