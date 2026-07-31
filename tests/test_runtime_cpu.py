from __future__ import annotations

import pytest

from game_cls.runtime import CpuAccelerator, NpuAccelerator, build_runtime


def test_cpu_accelerator_device_and_spawn_policy() -> None:
    acc = CpuAccelerator()
    assert acc.device.type == "cpu"
    assert acc.requires_spawn_workers is False
    acc.setup(0)
    acc.synchronize()


def test_npu_accelerator_requires_spawn() -> None:
    assert NpuAccelerator().requires_spawn_workers is True


def test_build_runtime_cpu_single_process() -> None:
    class _Sel:
        pass

    acc = _Sel()
    acc.type = "cpu"
    dist = _Sel()
    dist.type = "single_process"
    dist.params = {}
    runtime_sel = _Sel()
    runtime_sel.accelerator = acc
    runtime_sel.distributed = dist

    rt = build_runtime(runtime_sel)
    assert rt.accelerator.device.type == "cpu"
    assert rt.distributed.world_size == 1


def test_build_runtime_rejects_unknown_accelerator() -> None:
    class _Sel:
        pass

    acc = _Sel()
    acc.type = "quantum"
    dist = _Sel()
    dist.type = "single_process"
    dist.params = {}
    runtime_sel = _Sel()
    runtime_sel.accelerator = acc
    runtime_sel.distributed = dist

    with pytest.raises(ValueError, match="Unsupported accelerator"):
        build_runtime(runtime_sel)
