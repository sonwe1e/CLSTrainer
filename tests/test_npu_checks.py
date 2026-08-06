"""Unit tests for the NPU environment/operator probe (step4 P0-4).

The probe imports ``torch_npu`` lazily inside the call, so these tests
only need to patch ``sys.modules`` at call time — never re-import the
probe module (which would re-import torch and can crash on Windows).
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from game_cls.runtime.npu_checks import probe_npu_environment


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
