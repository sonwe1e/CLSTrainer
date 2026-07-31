from __future__ import annotations

from typing import Any


class SingleProcessDistributed:
    """Default non-distributed adapter. ``wrap_model`` is the identity."""

    def __init__(self, rank: int = 0, world_size: int = 1, local_rank: int = 0) -> None:
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def setup(self, backend: str) -> None:
        del backend

    def wrap_model(self, model: Any, device: Any) -> Any:
        return model.to(device)

    def barrier(self) -> None:
        pass

    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        del op
        return tensor

    def gather_object(self, value: Any, dst: int = 0) -> list[Any]:
        del dst
        return [value]

    def broadcast_object_list(self, object_list: list[Any], src: int = 0) -> None:
        del src
        # Single process: nothing to broadcast; object_list already holds the
        # local value at index 0.

    def cleanup(self) -> None:
        pass


class DdpDistributed:
    """Wraps PyTorch DistributedDataParallel behind the distributed facade."""

    def __init__(
        self,
        local_rank: int = 0,
        world_size: int = 1,
        rank: int = 0,
        *,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = False,
        gradient_as_bucket_view: bool = True,
    ) -> None:
        self._local_rank = local_rank
        self._world_size = world_size
        self._rank = rank
        self._backend = backend
        self._find_unused_parameters = find_unused_parameters
        self._broadcast_buffers = broadcast_buffers
        self._gradient_as_bucket_view = gradient_as_bucket_view

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def setup(self, backend: str) -> None:
        import torch.distributed as dist

        dist.init_process_group(backend=backend or self._backend, init_method="env://")

    def wrap_model(self, model: Any, device: Any) -> Any:
        from torch.nn.parallel import DistributedDataParallel

        # Move the model to the target device *before* wrapping it in DDP.
        # DDP does not move parameters itself; passing a model that lives on
        # the wrong device would raise a "model parameters are not on device"
        # error at the first forward/backward pass.
        model = model.to(device)
        return DistributedDataParallel(
            model,
            device_ids=[self._local_rank],
            find_unused_parameters=self._find_unused_parameters,
            broadcast_buffers=self._broadcast_buffers,
            gradient_as_bucket_view=self._gradient_as_bucket_view,
        )

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier()

    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        import torch.distributed as dist

        reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.from_str(op)  # type: ignore[attr-defined]
        dist.all_reduce(tensor, op=reduce_op)
        return tensor

    def gather_object(self, value: Any, dst: int = 0) -> list[Any]:
        import torch.distributed as dist

        if self._world_size <= 1:
            return [value]
        results = [None for _ in range(self._world_size)] if self._rank == dst else None
        dist.gather_object(value, results, dst=dst)
        # ``results`` is ``None`` on non-dst ranks; return an empty list so
        # callers can treat the return value uniformly.
        return results if results is not None else []

    def broadcast_object_list(self, object_list: list[Any], src: int = 0) -> None:
        import torch.distributed as dist

        if self._world_size <= 1:
            return
        dist.broadcast_object_list(object_list, src=src)

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
