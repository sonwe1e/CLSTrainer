from __future__ import annotations

import torch


def main() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"torch_npu import failed: {exc}") from exc

    available = bool(torch.npu.is_available())  # type: ignore[attr-defined]
    print("torch       :", torch.__version__)
    print("torch_npu   :", getattr(torch_npu, "__version__", "unknown"))
    print("npu available:", available)
    print("device count :", torch.npu.device_count() if available else 0)  # type: ignore[attr-defined]
    if not available:
        raise SystemExit(1)

    torch.npu.set_device(0)  # type: ignore[attr-defined]
    device = torch.device("npu:0")
    x = torch.randn(32, 32, device=device)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        y = x @ x
    torch.npu.synchronize()  # type: ignore[attr-defined]
    print("BF16 matmul   : OK", tuple(y.shape), y.dtype)


if __name__ == "__main__":
    main()
