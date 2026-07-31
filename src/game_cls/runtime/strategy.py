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
        self._is_setup = False

    @property
    def accelerator(self) -> AcceleratorAdapter:
        return self._accelerator

    @property
    def distributed(self) -> DistributedAdapter:
        return self._distributed

    def setup(self) -> None:
        if self._is_setup:
            return
        self._accelerator.setup(self._distributed.local_rank)
        # The distributed adapter initializes its process group here (e.g.
        # ``dist.init_process_group`` for DDP). The adapter falls back to its
        # own configured backend when ``None`` is passed.
        try:
            self._distributed.setup(None)
        except Exception:
            # distributed.setup failed (e.g. init_process_group raised). The
            # accelerator is already set up, so mark the strategy as setup
            # anyway so that cleanup() will tear down both halves. Without
            # this, cleanup() would early-return and leak the accelerator state
            # (and any partial process-group state).
            self._is_setup = True
            raise
        self._is_setup = True

    def wrap_model(self, model: Any) -> Any:
        return self._distributed.wrap_model(model, self._accelerator.device)

    def backward(self, loss: Any, scaler: Any) -> None:
        """Backward pass with AMP scaling."""
        scaler.scale(loss).backward()

    def unscale_gradients(self, optimizer: Any, scaler: Any) -> None:
        """Unscale gradients before clipping."""
        scaler.unscale_(optimizer)

    def clip_gradients(self, parameters: Any, max_norm: float) -> Any:
        """Clip gradients and return the grad norm."""
        import torch

        return torch.nn.utils.clip_grad_norm_(
            [p for p in parameters if p.requires_grad], max_norm
        )

    def optimizer_step(self, optimizer: Any, scaler: Any) -> None:
        """Step the optimizer and update the scaler."""
        scaler.step(optimizer)
        scaler.update()

    def autocast(self, enabled: bool, dtype: str):
        return self._accelerator.autocast(enabled, dtype)

    def barrier(self) -> None:
        self._distributed.barrier()

    def synchronize(self) -> None:
        self._accelerator.synchronize()

    def cleanup(self) -> None:
        if not self._is_setup:
            return
        self._distributed.cleanup()
        self._is_setup = False
