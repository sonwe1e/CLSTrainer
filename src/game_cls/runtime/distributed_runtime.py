"""Unified distributed launch validation and runtime initialization.

Single place that validates environment variables, accelerator/backend
combinations, ranks and world size BEFORE any device or process group is
created, then initializes the device and c10d. Replaces the duplicated
device-init logic that used to live in ``engine/device.py`` and
``engine/distributed.py``.

The two checks that protect against silent corruption:

* ``WORLD_SIZE > 1`` with ``distributed.enabled=false`` (torchrun launch
  of a single-process config) would make every rank train independently
  and race on the same checkpoint/status/report files.
* A backend that does not match the accelerator (e.g. nccl on CPU) fails
  fast instead of hanging or crashing mid-training.
"""

from __future__ import annotations

import os

from game_cls.engine.device import initialize_device

# Valid accelerator -> process-group backend combinations. NPU uses hccl,
# CUDA uses nccl, CPU uses gloo.
_ACCELERATOR_BACKENDS: dict[str, tuple[str, ...]] = {
    "cpu": ("gloo",),
    "cuda": ("nccl",),
    "npu": ("hccl",),
}


def _launch_environment() -> dict[str, int]:
    return {
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
    }


def validate_launch_environment(config: dict) -> dict[str, int]:
    """Validate launch env vars and config; returns (rank, world_size,
    local_rank) facts.

    Raises RuntimeError on every inconsistency; nothing is initialized.
    """
    distributed_config = config.get("distributed", {}) or {}
    distributed = bool(distributed_config.get("enabled", False))
    accelerator = str(config["device"]["accelerator"])
    if accelerator == "auto":
        import torch

        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    backend = str(distributed_config.get("backend", "") or "")
    env = _launch_environment()
    world_size = env["world_size"]
    rank = env["rank"]
    local_rank = env["local_rank"]

    if world_size > 1 and not distributed:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} but distributed.enabled=false: every "
            "rank would train independently and race on the same "
            "checkpoint/status/report files. Either launch a single "
            "process (no torchrun) or set distributed.enabled=true."
        )
    if distributed and world_size <= 1:
        raise RuntimeError(
            "distributed.enabled=true requires WORLD_SIZE > 1; launch "
            "via torchrun (WORLD_SIZE/RANK/LOCAL_RANK are read from the "
            "environment)."
        )
    if distributed and not backend:
        raise RuntimeError(
            "distributed.enabled=true requires distributed.backend "
            f"(one of {_ACCELERATOR_BACKENDS.get(accelerator, ())})."
        )
    if not 0 <= rank < world_size:
        raise RuntimeError(
            f"RANK={rank} is outside [0, WORLD_SIZE={world_size})."
        )
    if not 0 <= local_rank < world_size:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside [0, WORLD_SIZE={world_size})."
        )
    allowed = _ACCELERATOR_BACKENDS.get(accelerator)
    if allowed is None:
        raise ValueError(f"Unsupported accelerator: {accelerator!r}")
    if backend and backend not in allowed:
        raise RuntimeError(
            f"accelerator={accelerator!r} requires a process-group backend "
            f"in {allowed}; got {backend!r}."
        )
    return env


def init_runtime(config: dict) -> tuple[int, int, int, object]:
    """Validate the launch, bind the device, then initialize c10d.

    Returns ``(rank, world_size, local_rank, device)``. Idempotent for the
    process-group part: repeated calls on an already-initialized process
    group are no-ops.
    """
    import torch.distributed as dist

    distributed_config = config.get("distributed", {}) or {}
    distributed = bool(distributed_config.get("enabled", False))
    env = validate_launch_environment(config)
    rank = env["rank"]
    local_rank = env["local_rank"]
    world_size = env["world_size"]
    device = initialize_device(config["device"]["accelerator"], local_rank)
    if distributed and not dist.is_initialized():
        dist.init_process_group(
            backend=str(distributed_config["backend"]),
            init_method="env://",
        )
    return rank, world_size, local_rank, device


def cleanup() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_initialized() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()
