"""Distributed launch validation (step3 P0-4)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch


def _config(**distributed: object) -> dict:
    return {
        "device": {"accelerator": "cpu"},
        "distributed": distributed or {"enabled": False},
    }


class LaunchEnvironmentValidationTests(unittest.TestCase):
    def _validate(self, config: dict) -> dict[str, int]:
        from game_cls.runtime.distributed_runtime import (
            validate_launch_environment,
        )

        return validate_launch_environment(config)

    def test_single_process_cpu_is_valid(self) -> None:
        with patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"},
            clear=False,
        ):
            facts = self._validate(_config(enabled=False))
        self.assertEqual(facts["world_size"], 1)

    def test_torchrun_launch_with_distributed_disabled_is_refused(self) -> None:
        # The reverse check: WORLD_SIZE>1 with distributed.enabled=false
        # would let every rank race on the same files.
        with patch.dict(
            os.environ,
            {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"},
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "WORLD_SIZE=2"):
            self._validate(_config(enabled=False))

    def test_distributed_enabled_without_torchrun_is_refused(self) -> None:
        with patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"},
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "WORLD_SIZE > 1"):
            self._validate(
                _config(enabled=True, backend="gloo")
            )

    def test_distributed_enabled_requires_backend(self) -> None:
        with patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "backend"):
            self._validate(_config(enabled=True))

    def test_backend_must_match_accelerator(self) -> None:
        # CUDA requires nccl; cpu requires gloo.
        with patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "nccl"):
            self._validate(_config(enabled=True, backend="nccl"))
        with patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
            clear=False,
        ):
            self._validate(_config(enabled=True, backend="gloo"))

    def test_cuda_npu_backend_matrix(self) -> None:
        from game_cls.runtime.distributed_runtime import (
            _ACCELERATOR_BACKENDS,
        )

        self.assertEqual(_ACCELERATOR_BACKENDS["cuda"], ("nccl",))
        self.assertEqual(_ACCELERATOR_BACKENDS["npu"], ("hccl",))
        self.assertEqual(_ACCELERATOR_BACKENDS["cpu"], ("gloo",))

    def test_out_of_range_rank_is_refused(self) -> None:
        with patch.dict(
            os.environ,
            {"RANK": "5", "LOCAL_RANK": "5", "WORLD_SIZE": "2"},
            clear=False,
        ), self.assertRaisesRegex(RuntimeError, "RANK=5"):
            self._validate(_config(enabled=True, backend="gloo"))

    def test_unsupported_accelerator_is_refused(self) -> None:
        with patch.dict(
            os.environ,
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"},
            clear=False,
        ), self.assertRaisesRegex(ValueError, "accelerator"):
            self._validate(
                {
                    "device": {"accelerator": "tpu"},
                    "distributed": {"enabled": False},
                }
            )


if __name__ == "__main__":
    unittest.main()
