"""Unit tests for the NPU environment/operator probe (step4 P0-4).

The probe imports ``torch_npu`` lazily inside the call, so these tests
only need to patch ``sys.modules`` at call time — never re-import the
probe module (which would re-import torch and can crash on Windows).
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from game_cls.runtime.npu_checks import (
    _DEVICE_OP_PROBES,
    _describe_probe_failure,
    probe_npu_environment,
)


def test_probe_returns_warn_when_torch_npu_absent() -> None:
    had_torch_npu = "torch_npu" in sys.modules
    saved = sys.modules.get("torch_npu")
    sys.modules.pop("torch_npu", None)
    try:
        result = probe_npu_environment()
    finally:
        if had_torch_npu:
            sys.modules["torch_npu"] = saved

    assert "torch_npu available" in result
    ok, detail = result["torch_npu available"]
    assert ok is None  # WARN, not FAIL
    assert "not installed" in detail
    # No device-operator probes can run without torch_npu.
    assert not any("device op:" in key for key in result)


def test_probe_fails_when_torch_npu_installed_but_device_unavailable() -> None:
    import torch

    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.__version__ = "2.1.0"
    # Force torch.npu.is_available() -> False (create=True works even when
    # the CPU build has no torch.npu), and only swap torch_npu in sys.modules.
    with (
        patch.object(
            torch,
            "npu",
            SimpleNamespace(is_available=lambda: False),
            create=True,
        ),
        patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
    ):
        result = probe_npu_environment()
    ok, detail = result["torch_npu available"]
    assert ok is False
    assert "not available" in detail


def test_grad_scaler_probe_runs_a_real_amp_step() -> None:
    """The probe must pass on a working device.

    It used to call ``.backward()`` on a tensor with no ``requires_grad``,
    so autograd raised before the scaler was exercised and doctor reported
    ``device op: GradScaler [FAIL]`` on every host.
    """
    import torch

    _DEVICE_OP_PROBES["GradScaler"](torch.device("cpu"))


def test_grad_scaler_probe_catches_a_skipped_step() -> None:
    """A scaler whose step() never applies the update must be reported."""
    import torch
    import torch.amp

    class _SkippingScaler(torch.amp.GradScaler):
        def step(self, optimizer, *args, **kwargs):  # type: ignore[override]
            return None  # pretend every step was rejected as non-finite

    with (
        patch.object(torch.amp, "GradScaler", _SkippingScaler),
        pytest.raises(RuntimeError, match="did not apply the update"),
    ):
        _DEVICE_OP_PROBES["GradScaler"](torch.device("cpu"))


def test_every_device_op_probe_passes_on_cpu() -> None:
    """CPU stands in for a healthy device: no probe may fail on it."""
    import torch

    for name, probe in _DEVICE_OP_PROBES.items():
        try:
            probe(torch.device("cpu"))
        except Exception as exc:  # noqa: BLE001 - report which probe broke
            raise AssertionError(f"probe {name!r} failed on CPU: {exc}") from exc


def test_probe_failure_detail_keeps_the_cause_chain() -> None:
    """CANN puts the diagnosis past 200 chars and behind a wrapper."""
    inner = RuntimeError("EZ9999: op[Cast] failed, " + "detail " * 60)
    try:
        try:
            raise inner
        except RuntimeError as exc:
            raise ValueError("run failed") from exc
    except ValueError as exc:
        detail = _describe_probe_failure(exc)

    assert "ValueError: run failed" in detail
    assert "EZ9999" in detail
    assert "caused by" in detail
    assert len(detail) > 200
