from __future__ import annotations

import os


def initialize_runtime(config: dict):
    """Register the accelerator, bind the device, then initialize c10d."""
    import torch
    import torch.distributed as dist

    distributed_config = config.get("distributed", {})
    distributed = bool(distributed_config.get("enabled", False))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    accelerator = config["device"]["accelerator"]
    if accelerator == "auto":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    if accelerator == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU training requires a preinstalled torch_npu") from exc
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif accelerator == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported accelerator: {accelerator}")

    if distributed:
        if world_size <= 1:
            raise RuntimeError("Distributed mode requires WORLD_SIZE greater than one")
        dist.init_process_group(
            backend=distributed_config["backend"],
            init_method="env://",
        )
    return rank, world_size, local_rank, device


def distributed_context(config: dict) -> tuple[int, int, int]:
    """Compatibility helper for callers that do not initialize a device."""
    rank, world_size, local_rank, _ = initialize_runtime(
        {"distributed": config, "device": {"accelerator": "cpu"}}
    )
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_distributed() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()
