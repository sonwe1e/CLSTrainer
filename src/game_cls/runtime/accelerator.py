from __future__ import annotations

from contextlib import nullcontext
from typing import Any


class _BaseAccelerator:
    def make_grad_scaler(self, enabled: bool, dtype: str) -> Any:
        import torch

        use_fp16 = enabled and dtype == "float16"
        return torch.amp.GradScaler(device=self.device.type, enabled=use_fp16)


class CpuAccelerator(_BaseAccelerator):
    @property
    def device(self) -> Any:
        import torch

        return torch.device("cpu")

    @property
    def requires_spawn_workers(self) -> bool:
        return False

    def setup(self, local_rank: int) -> None:
        del local_rank

    def autocast(self, enabled: bool, dtype: str):
        del dtype
        return nullcontext() if enabled else nullcontext()

    def synchronize(self) -> None:
        pass


class CudaAccelerator(_BaseAccelerator):
    def __init__(self, local_rank: int = 0) -> None:
        self._local_rank = local_rank

    @property
    def device(self) -> Any:
        import torch

        return torch.device(f"cuda:{self._local_rank}")

    @property
    def requires_spawn_workers(self) -> bool:
        return False

    def setup(self, local_rank: int) -> None:
        import torch

        self._local_rank = local_rank
        torch.cuda.set_device(local_rank)

    def autocast(self, enabled: bool, dtype: str):
        import torch

        if not enabled:
            return nullcontext()
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=torch_dtype)

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize(self.device)


class NpuAccelerator(_BaseAccelerator):
    def __init__(self, local_rank: int = 0) -> None:
        self._local_rank = local_rank

    @property
    def device(self) -> Any:
        import torch

        return torch.device(f"npu:{self._local_rank}")

    @property
    def requires_spawn_workers(self) -> bool:
        return True

    def setup(self, local_rank: int) -> None:
        import torch

        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU training requires a preinstalled torch_npu") from exc
        self._local_rank = local_rank
        torch.npu.set_device(local_rank)

    def autocast(self, enabled: bool, dtype: str):
        import torch

        if not enabled:
            return nullcontext()
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type="npu", dtype=torch_dtype)

    def synchronize(self) -> None:
        import torch

        torch.npu.synchronize(self.device)
