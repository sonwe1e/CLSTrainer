"""DataLoader throughput probe (step5 P5).

Compares ``[png, packed_uint8] x [num_workers] x [prefetch_factor]`` by
building the exact production training DataLoader (``_make_dataloaders``)
for each variant and measuring ``samples/s`` over a fixed number of steps.
The worker/prefetch/packed decision is made by measurement, never by
hard-coded defaults.
"""

from __future__ import annotations

import copy
import time
from typing import Any

_LOADER_VARIANTS = [
    (backend, workers, prefetch)
    for backend in ("png", "packed_uint8")
    for workers in (0, 4)
    for prefetch in (1, 2, 4)
]


def run_data_probe(
    config: dict,
    *,
    steps: int = 20,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Measure the loader variants and return ``{label: metrics}``."""
    from game_cls.engine.training.loaders import _make_dataloaders

    results: dict[str, Any] = {}
    for backend, workers, prefetch in _LOADER_VARIANTS:
        probe_config = copy.deepcopy(config)
        probe_config["data"]["backend"] = backend
        probe_config["dataloader"]["train"] = {
            **(probe_config["dataloader"].get("train") or {}),
            "num_workers": workers,
            "prefetch_factor": prefetch,
        }
        if workers == 0:
            probe_config["dataloader"]["train"].pop(
                "multiprocessing_context", None
            )
        else:
            probe_config["dataloader"]["train"].setdefault(
                "multiprocessing_context",
                "spawn" if config["device"].get("accelerator") == "npu" else "fork",
            )
        probe_config["train"]["local_batch_size"] = batch_size
        try:
            bundle = _make_dataloaders(probe_config, rank=0, world_size=1)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError):
            # Missing indexes / audit for this backend: skip the variant.
            continue
        start = time.perf_counter()
        sample_count = 0
        for index, batch in enumerate(bundle.train):
            if index >= steps:
                break
            sample_count += int(batch["images"].shape[0])
        elapsed = max(time.perf_counter() - start, 1e-9)
        results[f"{backend}_w{workers}_p{prefetch}"] = {
            "backend": backend,
            "num_workers": workers,
            "prefetch_factor": prefetch,
            "steps": steps,
            "samples_per_second": round(sample_count / elapsed, 1),
            "batch_samples": sample_count,
            "elapsed_seconds": round(elapsed, 3),
        }
    return results
