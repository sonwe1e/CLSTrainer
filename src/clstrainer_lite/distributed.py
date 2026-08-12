from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Runtime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _resolve_accelerator(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
        try:
            if torch.npu.is_available():  # type: ignore[attr-defined]
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_backend(accelerator: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[accelerator]


def init_runtime(config: dict[str, Any]) -> Runtime:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accelerator = _resolve_accelerator(str(config.get("accelerator", "auto")))
    backend = _resolve_backend(accelerator, str(config.get("backend", "auto")))

    if not 0 <= rank < world_size:
        raise RuntimeError(f"RANK={rank} is outside WORLD_SIZE={world_size}")

    if accelerator == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU mode requires a compatible torch_npu installation") from exc
        torch.npu.set_device(local_rank)  # type: ignore[attr-defined]
        device = torch.device(f"npu:{local_rank}")
        if backend != "hccl":
            raise RuntimeError("NPU distributed training must use backend=hccl")
    elif accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if backend != "nccl" and world_size > 1:
            raise RuntimeError("CUDA distributed training should use backend=nccl")
    elif accelerator == "cpu":
        device = torch.device("cpu")
        if backend != "gloo" and world_size > 1:
            raise RuntimeError("CPU distributed training must use backend=gloo")
    else:
        raise ValueError(f"Unsupported accelerator: {accelerator}")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return Runtime(rank, local_rank, world_size, device, backend if world_size > 1 else None)


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu") and callable(getattr(torch.npu, "manual_seed_all", None)):
        try:
            torch.npu.manual_seed_all(seed)  # type: ignore[attr-defined]
        except Exception:
            pass


def wrap_ddp(
    model: torch.nn.Module,
    runtime: Runtime,
    *,
    find_unused_parameters: bool = True,
) -> torch.nn.Module:
    if not runtime.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel

    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": bool(find_unused_parameters),
    }
    if runtime.device.type in {"cuda", "npu"}:
        kwargs["device_ids"] = [runtime.local_rank]
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
