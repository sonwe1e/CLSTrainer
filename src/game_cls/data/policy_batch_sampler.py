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

from .contracts.data import SamplingContext, SamplingPolicy


class PolicyBatchSampler:
    """Adapts a :class:`SamplingPolicy` to PyTorch's ``BatchSampler`` interface.

    The caller provides a catalog (the video/sequence index) and a policy. Each
    step, the sampler asks the policy for a rank-local batch via
    ``sample_rank_batch``.
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
        """Set the epoch (and optionally the starting step for resume)."""
        self._epoch = epoch
        self._step = start_step

    def __iter__(self):
        for _ in range(self._steps_per_epoch):
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
        return self._steps_per_epoch

    def state_dict(self) -> dict:
        return self._policy.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._policy.load_state_dict(state)
