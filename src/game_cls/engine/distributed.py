"""Compatibility shim over ``game_cls.runtime.distributed_runtime``.

All validation and initialization logic lives in the unified
``DistributedRuntime`` module; these helpers keep existing imports
working.
"""

from __future__ import annotations

from game_cls.runtime import distributed_runtime

initialize_runtime = distributed_runtime.init_runtime
distributed_barrier = distributed_runtime.barrier
cleanup_distributed = distributed_runtime.cleanup
is_distributed = distributed_runtime.is_initialized
validate_launch_environment = distributed_runtime.validate_launch_environment

__all__ = [
    "initialize_runtime",
    "distributed_barrier",
    "cleanup_distributed",
    "is_distributed",
    "validate_launch_environment",
]


def distributed_context(config: dict) -> tuple[int, int, int]:
    """Compatibility helper for callers that do not initialize a device."""
    rank, world_size, local_rank, _ = initialize_runtime(
        {"distributed": config, "device": {"accelerator": "cpu"}}
    )
    return rank, world_size, local_rank
