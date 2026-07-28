from __future__ import annotations

import os


def distributed_context(config: dict) -> tuple[int, int, int]:
    enabled = config.get("enabled", False)
    if not enabled:
        return 0, 1, 0
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend=config["backend"], init_method="env://")
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
