from __future__ import annotations

from contextlib import nullcontext


def initialize_device(accelerator: str, local_rank: int = 0):
    import torch

    if accelerator == "auto":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if accelerator == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU training requires torch_npu") from exc
        torch.npu.set_device(local_rank)  # type: ignore[attr-defined]
        return torch.device(f"npu:{local_rank}")
    if accelerator == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported accelerator: {accelerator}")


def autocast_context(device, enabled: bool, dtype_name: str):
    import torch

    if not enabled or device.type == "cpu":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type=device.type, dtype=dtype)
