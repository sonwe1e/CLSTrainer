from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from game_cls.engine.checkpoint import (
    capture_random_state,
    save_checkpoint_pair,
)
from game_cls.engine.distributed import (
    is_distributed,
)


def _distributed_sum_int(value: int) -> int:
    if not is_distributed():
        return value
    import torch.distributed as dist

    values: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return sum(int(item) for item in values)


def _schedule_factor(config: dict, total_steps: int, base_lr: float, step: int) -> float:
    """The LambdaLR multiplier at ``step``.

    Shared by ``_build_scheduler`` and the staged-unfreeze boundary so the
    rebuilt scheduler can be positioned continuously: the unfreeze path must
    place every new parameter group at the LR the schedule would have produced
    at ``global_step`` (audit P1-3).
    """
    warmup = int(config.get("warmup_steps", 0))
    min_lr = float(config.get("min_learning_rate", 0.0))
    min_ratio = min_lr / base_lr if base_lr else 0.0
    if warmup > 0 and step < warmup:
        return max(1e-8, (step + 1) / warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cosine


def _build_scheduler(optimizer, config: dict, total_steps: int):
    import torch

    base_lr = max(group["lr"] for group in optimizer.param_groups)

    def factor(step: int) -> float:
        return _schedule_factor(config, total_steps, base_lr, step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _broadcast_object(value, rank: int):
    if not is_distributed():
        return value
    import torch.distributed as dist

    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _gather_random_states(rank: int, world_size: int) -> list[dict] | None:
    local = capture_random_state()
    if not is_distributed():
        return [local]
    import torch.distributed as dist

    gathered: list[Any] | None = (
        [None for _ in range(world_size)] if rank == 0 else None
    )
    dist.gather_object(local, gathered, dst=0)
    return gathered


def _normalized_position(
    epoch: int, step_in_epoch: int, steps_per_epoch: int
) -> tuple[int, int]:
    if step_in_epoch >= steps_per_epoch:
        return epoch + 1, 0
    return epoch, step_in_epoch


def _save_all_ranks(
    *,
    output_dir: Path,
    tag: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_metrics: dict,
    config: dict,
    sampler,
    evaluation_state: dict,
    rank: int,
    world_size: int,
    force_full_model: bool = False,
) -> None:
    states = _gather_random_states(rank, world_size)
    save_epoch, save_step = _normalized_position(
        epoch, step_in_epoch, int(config["train"]["steps_per_epoch"])
    )
    if rank == 0:
        sampler_state = sampler.state_dict(save_step)
        sampler_state["epoch"] = save_epoch
        checkpoint_cfg = config["checkpoint"]
        state_mode = checkpoint_cfg.get("periodic_state_mode", "full")
        full_model_every = int(checkpoint_cfg.get("full_model_every_steps", 0))
        write_model_only = (
            force_full_model
            or state_mode == "full"
            or (full_model_every > 0 and global_step % full_model_every == 0)
        )
        save_checkpoint_pair(
            output_dir / "checkpoints",
            tag,
            model,
            optimizer,
            scheduler,
            scaler,
            save_epoch,
            global_step,
            best_metrics,
            config,
            step_in_epoch=save_step,
            sampler_state=sampler_state,
            rank_random_states=states,
            evaluation_state=evaluation_state,
            state_mode=state_mode,
            write_model_only=write_model_only,
        )
