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


def capture_random_state() -> dict:
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    npu = getattr(torch, "npu", None)
    if npu is not None and callable(getattr(npu, "is_available", None)):
        if npu.is_available() and hasattr(npu, "get_rng_state_all"):
            state["npu"] = npu.get_rng_state_all()
    return state


def restore_random_state(state: dict) -> None:
    import numpy as np
    import torch

    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    npu = getattr(torch, "npu", None)
    if (
        state.get("npu") is not None
        and npu is not None
        and hasattr(npu, "set_rng_state_all")
    ):
        npu.set_rng_state_all(state["npu"])


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
    step_in_epoch: int = 0,
    sampler_state: dict | None = None,
    rank_random_states: list[dict] | None = None,
    evaluation_state: dict | None = None,
) -> None:
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
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "sampler_epoch": epoch,
            "sampler_state": sampler_state or {
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
            },
            "best_metrics": best_metrics,
            "random_state": (
                rank_random_states[0]
                if rank_random_states
                else capture_random_state()
            ),
            "rank_random_states": rank_random_states,
            "evaluation_state": evaluation_state or {},
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
