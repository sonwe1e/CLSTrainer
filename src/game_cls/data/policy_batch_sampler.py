"""Glue between :class:`SamplingPolicy` and PyTorch's ``BatchSampler`` interface.

PyTorch's ``DataLoader`` expects a ``BatchSampler`` that yields a batch of
indices (or, in our case, a list of ``SampleRequest``-like dicts) per step.
The :class:`SamplingPolicy` protocol produces exactly that, so this class
adapts the policy to the ``BatchSampler`` interface.

This is the D3 migration step (USERPLAN §11): the ``DataLoader`` now depends
only on :class:`PolicyBatchSampler`, which delegates to the configured
:class:`SamplingPolicy`. New sampling algorithms are added by implementing the
policy protocol — the training loop and DataModule never branch on the
algorithm name.
"""
from __future__ import annotations

from typing import Any

from ..contracts.data import SamplingContext, SamplingPolicy


class PolicyBatchSampler:
    """Adapts a :class:`SamplingPolicy` to PyTorch's ``BatchSampler`` interface.

    This is the D3 migration glue (USERPLAN §11): the ``DataLoader`` depends
    only on this class, which delegates to the configured
    :class:`SamplingPolicy`. New sampling algorithms are added by implementing
    the policy protocol — the training loop and DataModule never branch on the
    algorithm name.

    The wrapper exposes the full ``BatchSampler`` protocol (``__iter__``,
    ``__len__``, ``set_epoch``, ``state_dict``, ``load_state_dict``) by
    delegating to the underlying policy. Policies that already implement the
    BatchSampler protocol (e.g. :class:`BalancedGameLabelDeltaPolicy`) are
    used directly; simpler policies that only implement ``sample_rank_batch``
    are driven step-by-step by this wrapper.
    """

    def __init__(
        self,
        policy: SamplingPolicy,
        catalog: Any,
        steps_per_epoch: int,
    ) -> None:
        self._policy = policy
        self._catalog = catalog
        self._steps_per_epoch = steps_per_epoch
        self._epoch = 0
        self._step = 0

    @property
    def policy(self) -> SamplingPolicy:
        return self._policy

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        """Set the epoch (and optionally the starting step for resume).

        If the underlying policy implements ``set_epoch`` (i.e. it is itself a
        full BatchSampler), delegate to it. Otherwise, track the epoch/step
        internally for the step-by-step iteration path.
        """
        self._epoch = epoch
        self._step = start_step
        if hasattr(self._policy, "set_epoch"):
            # Inspect the policy's signature so we only pass start_step when
            # the policy actually accepts it. This avoids silently swallowing
            # TypeErrors raised inside the policy's own set_epoch body.
            import inspect

            try:
                sig = inspect.signature(self._policy.set_epoch)
            except (TypeError, ValueError):
                sig = None
            if sig is not None and "start_step" in sig.parameters:
                self._policy.set_epoch(epoch, start_step=start_step)
            else:
                self._policy.set_epoch(epoch)

    def __iter__(self):
        # If the policy already implements the BatchSampler protocol (has its
        # own ``__iter__`` that yields batches), delegate to it. This preserves
        # the exact-resume semantics of policies like
        # ``BalancedGameLabelDeltaPolicy`` that know how to skip to ``start_step``.
        if hasattr(self._policy, "__iter__") and hasattr(self._policy, "__len__"):
            yield from self._policy
            return
        # Otherwise, drive the policy step-by-step via ``sample_rank_batch``.
        # Start from ``self._step`` (which may be mid-epoch on resume) so that
        # resumed training sees the same data sequence.
        for _ in range(self._step, self._steps_per_epoch):
            context = SamplingContext(
                epoch=self._epoch,
                step=self._step,
                global_batch_size=0,  # unused by rank-local policies
                world_size=1,
                seed=0,
            )
            batch = self._policy.sample_rank_batch(self._catalog, context)
            yield batch
            self._step += 1

    def __len__(self) -> int:
        # Prefer the policy's own length when it implements BatchSampler.
        if hasattr(self._policy, "__len__"):
            try:
                return len(self._policy)
            except TypeError:
                pass
        return self._steps_per_epoch

    def state_dict(self, step_in_epoch: int | None = None) -> dict:
        # Delegate to the policy's state_dict, passing step_in_epoch when
        # the policy supports it (the legacy VideoBalancedPairBatchSampler
        # needs it to capture the sampler position for exact resume).
        try:
            return self._policy.state_dict(step_in_epoch)
        except TypeError:
            return self._policy.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._policy.load_state_dict(state)
