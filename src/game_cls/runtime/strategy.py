from __future__ import annotations

from typing import Any

from ..contracts.runtime import AcceleratorAdapter, DistributedAdapter, RuntimeStrategy


class ComposedRuntimeStrategy:
    """Composition of an accelerator and a distributed adapter.

    The trainer talks to this facade and never branches on accelerator name or
    distributed backend directly (USERPLAN §12.4).
    """

    def __init__(
        self,
        accelerator: AcceleratorAdapter,
        distributed: DistributedAdapter,
    ) -> None:
        self._accelerator = accelerator
        self._distributed = distributed

    @property
    def accelerator(self) -> AcceleratorAdapter:
        return self._accelerator

    @property
    def distributed(self) -> DistributedAdapter:
        return self._distributed

    def setup(self) -> None:
        self._accelerator.setup(self._distributed.local_rank)
        # The distributed adapter initializes its process group here (e.g.
        # ``dist.init_process_group`` for DDP). The adapter falls back to its
        # own configured backend when ``None`` is passed.
        self._distributed.setup(None)

    def wrap_model(self, model: Any) -> Any:
        return self._distributed.wrap_model(model, self._accelerator.device)

    def backward(self, loss: Any, scaler: Any) -> None:
        scaler.scale(loss).backward()

    def clip_gradients(self, parameters: Any, max_norm: float) -> Any:
        import torch

        return torch.nn.utils.clip_grad_norm_(
            [p for p in parameters if p.requires_grad], max_norm
        )

    def optimizer_step(self, optimizer: Any, scaler: Any) -> None:
        scaler.step(optimizer)
        scaler.update()

    def autocast(self, enabled: bool, dtype: str):
        return self._accelerator.autocast(enabled, dtype)

    def barrier(self) -> None:
        self._distributed.barrier()

    def synchronize(self) -> None:
        self._accelerator.synchronize()

    def cleanup(self) -> None:
        self._distributed.cleanup()
