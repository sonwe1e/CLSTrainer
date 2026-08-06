"""NPU environment and device-operator probes (step4 P0-4).

A small, self-contained set of checks that ``doctor`` runs when the
config requests the ``npu`` accelerator: torch_npu availability, CANN
version, HCCL backend visibility, BF16 autocast support, and a minimal
forward probe for the operators the training/evaluation path actually
uses (``bincount``, ``scatter_add_``, ``nonzero``, ``index_select``,
GradScaler). Each probe is independent so a single missing operator is
reported as its own failure instead of aborting the whole check.

All functions degrade gracefully on non-NPU hosts: they return a
``(ok, detail)`` pair where ``ok`` is ``None`` (unknown/skipped) when the
operator genuinely cannot be probed because torch_npu is absent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def probe_npu_environment() -> dict[str, tuple[bool | None, str]]:
    """Run the NPU checks and return ``{label: (ok, detail)}``.

    ``ok`` is ``True``/``False`` when a decision was possible and ``None``
    when the probe was skipped (torch_npu not installed, or the operator
    does not apply). Doctor renders False as ``[FAIL]`` and None as
    ``[WARN]``.
    """
    import torch

    results: dict[str, tuple[bool | None, str]] = {}

    try:
        import torch_npu  # type: ignore

        version = getattr(torch_npu, "__version__", "?")
        available = bool(
            getattr(torch, "npu", None) and torch.npu.is_available()  # type: ignore[attr-defined]
        )
        results["torch_npu available"] = (
            (True, f"torch_npu {version}")
            if available
            else (False, f"torch_npu {version} not available")
        )
    except ImportError as exc:
        results["torch_npu available"] = (None, f"torch_npu not installed: {exc}")
        return results

    # CANN version: exposed via torch_npu.npu.sys_info / version when present.
    can_ok: bool | None = None
    can_detail = ""
    try:
        if hasattr(torch_npu, "npu") and hasattr(torch_npu.npu, "get_version"):
            can_detail = str(torch_npu.npu.get_version())
            can_ok = bool(can_detail)
        else:
            can_detail = "torch_npu.npu.get_version() unavailable"
            can_ok = None
    except Exception as exc:  # noqa: BLE001 - probe must not raise
        can_detail = f"torch_npu.npu.get_version() raised: {exc}"
        can_ok = False
    results["CANN version"] = (can_ok, can_detail)

    # HCCL: the accelerator-specific backend must be visible to c10d.
    hccl_ok: bool | None = None
    hccl_detail = ""
    try:
        import torch.distributed as dist

        if hasattr(dist, "is_hccl_available"):
            hccl_ok = bool(dist.is_hccl_available())
            hccl_detail = f"dist.is_hccl_available()={hccl_ok}"
        else:
            hccl_detail = "torch.distributed.is_hccl_available() not present"
            hccl_ok = None
    except Exception as exc:  # noqa: BLE001
        hccl_detail = f"hccl probe raised: {exc}"
        hccl_ok = False
    results["HCCL available"] = (hccl_ok, hccl_detail)

    # torch.device("npu") requires torch_npu to register the device type, so
    # only construct it once the device is actually available.
    if getattr(torch, "npu", None) and torch.npu.is_available():  # type: ignore[attr-defined]
        device = torch.device("npu")
        # BF16 autocast must be usable on-device (deployment parity).
        bf_ok: bool | None = None
        bf_detail = ""
        try:
            x = torch.zeros(4, 4, device=device)
            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                y = x @ x
            bf_ok = bool(y.dtype == torch.bfloat16)
            bf_detail = f"bfloat16 autocast ok, dtype={y.dtype}"
        except Exception as exc:  # noqa: BLE001
            bf_ok = False
            bf_detail = f"bfloat16 autocast raised: {exc}"
        results["BF16 autocast"] = (bf_ok, bf_detail)

        for op, probe in _DEVICE_OP_PROBES.items():
            try:
                probe(device)
                results[f"device op: {op}"] = (True, "ok")
            except Exception as exc:  # noqa: BLE001
                results[f"device op: {op}"] = (False, _describe_probe_failure(exc))
    return results


def _describe_probe_failure(exc: BaseException) -> str:
    """Render a probe failure without discarding the diagnosis.

    CANN operator errors put the actionable part (``EZ9999``, the operator
    name, the suggested workaround) well past the first 200 characters, and
    they arrive wrapped, so the outer message is often just "run failed".
    Keep the exception type, the full chain, and a generous budget.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        parts.append(
            f"{type(current).__name__}: {message}"
            if message
            else type(current).__name__
        )
        current = current.__cause__ or current.__context__
    detail = " <- caused by ".join(parts)
    limit = 4000
    if len(detail) > limit:
        detail = detail[:limit] + f"... [truncated, {len(detail)} chars total]"
    return detail


def _probe_bincount(device) -> None:
    import torch

    counts = torch.bincount(torch.tensor([0, 1, 1, 2, 0, 3], device=device))
    assert counts.shape[0] >= 4


def _probe_scatter_add(device) -> None:
    import torch

    out = torch.zeros(4, dtype=torch.float32, device=device)
    idx = torch.tensor([0, 2, 2, 3], device=device)
    out.scatter_add_(0, idx, torch.ones(4, device=device))


def _probe_nonzero(device) -> None:
    import torch

    idx = torch.nonzero(torch.tensor([0, 1, 0, 1], device=device))
    assert idx.numel() == 2


def _probe_index_select(device) -> None:
    import torch

    src = torch.arange(8, device=device).reshape(2, 4)
    idx = torch.tensor([1, 0], device=device)
    assert src.index_select(0, idx).shape == (2, 4)


def _probe_grad_scaler(device) -> None:
    """Exercise the whole AMP step, not just ``scale()``.

    The old probe called ``scaler.scale(loss).backward()`` on a tensor that
    never had ``requires_grad``, so autograd refused before the scaler was
    tested at all -- it reported a failure on every host, including working
    ones. The parts that actually break on a new CANN build are the
    inf/nan check inside ``step()`` and the scale bookkeeping in
    ``update()``, so drive a real parameter through both and confirm the
    optimizer moved it.
    """
    import torch
    from torch.amp import GradScaler

    scaler = GradScaler(device.type, enabled=True)
    weight = torch.nn.Parameter(torch.randn(4, 4, device=device))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    before = weight.detach().clone()

    loss = (weight * weight).sum()
    scaler.scale(loss).backward()
    if weight.grad is None:
        raise RuntimeError("scaler.scale(loss).backward() produced no gradient")
    scaler.unscale_(optimizer)
    if not bool(torch.isfinite(weight.grad).all()):
        raise RuntimeError("scaler.unscale_ produced non-finite gradients")
    scaler.step(optimizer)
    scaler.update()

    scale = float(scaler.get_scale())
    if not (scale > 0.0 and scale == scale and scale != float("inf")):
        raise RuntimeError(f"scaler.update() left a non-finite scale: {scale}")
    if bool(torch.equal(weight.detach(), before)):
        # A broken inf/nan check silently skips every step, so training
        # would run to completion without the loss ever moving.
        raise RuntimeError(
            "scaler.step() did not apply the update (weights unchanged) -- "
            "the inf/nan check likely rejects finite gradients"
        )


_DEVICE_OP_PROBES: dict[str, Callable[..., Any]] = {
    "bincount": _probe_bincount,
    "scatter_add_": _probe_scatter_add,
    "nonzero": _probe_nonzero,
    "index_select": _probe_index_select,
    "GradScaler": _probe_grad_scaler,
}
