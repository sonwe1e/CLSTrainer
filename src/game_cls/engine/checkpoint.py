from __future__ import annotations

import os
from pathlib import Path
import random
import hashlib
from functools import lru_cache
import shutil


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


def clone_checkpoint_pair(
    output_dir: str | Path, source_tag: str, target_tag: str
) -> None:
    """Create best aliases from an already serialized checkpoint pair."""
    output_dir = Path(output_dir)
    for prefix in ("model", "checkpoint"):
        source = output_dir / f"{prefix}_{source_tag}.pth"
        if not source.is_file():
            continue
        target = output_dir / f"{prefix}_{target_tag}.pth"
        temporary = target.with_suffix(target.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, target)


@lru_cache(maxsize=8)
def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    state_mode: str = "full",
    write_model_only: bool = True,
) -> None:
    output_dir = Path(output_dir)
    unwrapped = unwrap_model(model)
    full_state_dict = unwrapped.state_dict()
    if state_mode == "trainable_only":
        trainable_names = {
            name for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        }
        checkpoint_state_dict = {
            key: value
            for key, value in full_state_dict.items()
            if key in trainable_names
            or any(key.startswith(name.rsplit(".", 1)[0] + ".") for name in trainable_names)
        }
    elif state_mode == "full":
        checkpoint_state_dict = full_state_dict
    else:
        raise ValueError(f"Unsupported checkpoint state mode: {state_mode}")
    if write_model_only:
        _atomic_torch_save(full_state_dict, output_dir / f"model_{tag}.pth")
    base_checkpoint = config.get("model", {}).get("checkpoint_path")
    base_hash = (
        _file_sha256(str(Path(base_checkpoint).resolve()))
        if state_mode == "trainable_only"
        and base_checkpoint
        and Path(base_checkpoint).is_file()
        else None
    )
    _atomic_torch_save(
        {
            "model": checkpoint_state_dict,
            "model_state_mode": state_mode,
            "base_checkpoint": base_checkpoint,
            "base_checkpoint_sha256": base_hash,
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
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    expected_base_checkpoint: str | Path | None = None,
) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_mode = checkpoint.get("model_state_mode", "full")
    if state_mode == "trainable_only":
        stored_hash = checkpoint.get("base_checkpoint_sha256")
        if stored_hash and expected_base_checkpoint:
            actual_hash = _file_sha256(
                str(Path(expected_base_checkpoint).resolve())
            )
            if actual_hash != stored_hash:
                raise RuntimeError(
                    "Resume base checkpoint hash does not match the checkpoint state."
                )
        result = unwrap_model(model).load_state_dict(
            checkpoint["model"], strict=False
        )
        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected trainable checkpoint keys: {result.unexpected_keys}"
            )
    else:
        unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
