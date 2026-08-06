from __future__ import annotations

import random


def _tb_write_scalars(writer, prefix: str, payload: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize_device_for_metrics(device) -> None:
    import torch

    if device.type == "npu":
        torch.npu.synchronize()  # type: ignore[attr-defined]
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _initialize_data_worker(
    worker_id: int,
    *,
    num_threads: int,
) -> None:
    import torch

    # Keep worker-level CPU parallelism from multiplying across DataLoader
    # processes. This function must remain at module scope for spawn pickling.
    del worker_id
    torch.set_num_threads(max(1, int(num_threads)))
