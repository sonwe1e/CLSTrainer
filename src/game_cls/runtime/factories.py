from __future__ import annotations

import os
from typing import Any

from ..contracts.runtime import AcceleratorAdapter, DistributedAdapter, RuntimeStrategy
from .accelerator import CpuAccelerator, CudaAccelerator, NpuAccelerator
from .distributed import DdpDistributed, SingleProcessDistributed
from .strategy import ComposedRuntimeStrategy


def _resolve_accelerator_type(accelerator_type: str) -> str:
    if accelerator_type == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return accelerator_type


def _build_accelerator(accelerator_type: str, local_rank: int) -> AcceleratorAdapter:
    resolved = _resolve_accelerator_type(accelerator_type)
    if resolved == "cpu":
        return CpuAccelerator()
    if resolved == "cuda":
        return CudaAccelerator(local_rank)
    if resolved == "npu":
        return NpuAccelerator(local_rank)
    raise ValueError(f"Unsupported accelerator: {accelerator_type}")


def _build_distributed(
    distributed_type: str,
    distributed_params: dict[str, Any],
    rank: int,
    world_size: int,
    local_rank: int,
) -> DistributedAdapter:
    if distributed_type == "single_process":
        return SingleProcessDistributed(rank=rank, world_size=world_size, local_rank=local_rank)
    if distributed_type == "ddp":
        return DdpDistributed(
            local_rank=local_rank,
            world_size=world_size,
            rank=rank,
            backend=str(distributed_params.get("backend", "nccl")),
            find_unused_parameters=bool(distributed_params.get("find_unused_parameters", False)),
            broadcast_buffers=bool(distributed_params.get("broadcast_buffers", False)),
            gradient_as_bucket_view=bool(distributed_params.get("gradient_as_bucket_view", True)),
        )
    raise ValueError(f"Unsupported distributed type: {distributed_type}")


def build_runtime(
    runtime_config: Any,
    *,
    distributed_override: bool | None = None,
) -> RuntimeStrategy:
    """Build a :class:`RuntimeStrategy` from the ``runtime`` config section.

    ``distributed_override`` lets callers force single-process mode (e.g. tests)
    regardless of the config. When ``None``, the config's ``distributed.type``
    decides.
    """
    accelerator_type = str(runtime_config.accelerator.type)
    distributed_type = str(runtime_config.distributed.type)
    distributed_params = dict(runtime_config.distributed.params) if runtime_config.distributed.params else {}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if distributed_override is True:
        distributed_type = "ddp"
    elif distributed_override is False:
        distributed_type = "single_process"

    accelerator = _build_accelerator(accelerator_type, local_rank)
    distributed = _build_distributed(
        distributed_type, distributed_params, rank, world_size, local_rank
    )
    return ComposedRuntimeStrategy(accelerator, distributed)
