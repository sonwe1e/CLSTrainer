from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def test_npu_extension_and_device_are_ready_before_hccl_init():
    events = []
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.npu = SimpleNamespace(
        set_device=lambda rank: events.append(("set_device", rank))
    )
    fake_torch.device = lambda value: value
    fake_dist = ModuleType("torch.distributed")
    fake_dist.init_process_group = lambda **kwargs: events.append(
        ("init_process_group", kwargs["backend"])
    )
    fake_dist.is_initialized = lambda: False
    fake_torch.distributed = fake_dist
    fake_torch_npu = ModuleType("torch_npu")
    environment = {"LOCAL_RANK": "1", "RANK": "1", "WORLD_SIZE": "8"}
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torch.distributed": fake_dist,
            "torch_npu": fake_torch_npu,
        },
    ), patch.dict(os.environ, environment, clear=False):
        from game_cls.engine.distributed import initialize_runtime

        rank, world_size, local_rank, device = initialize_runtime(
            {
                "device": {"accelerator": "npu"},
                "distributed": {"enabled": True, "backend": "hccl"},
            }
        )
    assert (rank, world_size, local_rank, device) == (1, 8, 1, "npu:1")
    assert events == [("set_device", 1), ("init_process_group", "hccl")]

