from __future__ import annotations

import unittest

from game_cls.engine.training.config_validation import _validate_dataloader_config
from game_cls.engine.training.loaders import _loader_common


class DataLoaderPolicyTests(unittest.TestCase):
    def test_npu_loader_defaults_to_spawn(self) -> None:
        config = {
            "device": {"accelerator": "npu"},
            "dataloader": {
                "train": {
                    "num_workers": 2,
                    "persistent_workers": True,
                }
            },
        }

        options = _loader_common(config, "train")

        self.assertEqual(options["multiprocessing_context"], "spawn")
        self.assertEqual(options["timeout"], 180)
        self.assertTrue(options["persistent_workers"])
        self.assertEqual(options["prefetch_factor"], 2)
        self.assertIsNotNone(options["worker_init_fn"])

    def test_npu_loader_rejects_fork(self) -> None:
        config = {
            "device": {"accelerator": "npu"},
            "dataloader": {
                "multiprocessing_context": "fork",
                "train": {"num_workers": 1},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "must use"):
            _loader_common(config, "train")
        with self.assertRaisesRegex(RuntimeError, "must use spawn"):
            _validate_dataloader_config(config)

    def test_role_configuration_uses_scoped_worker_values(self) -> None:
        config = {
            "device": {"accelerator": "cpu"},
            "dataloader": {
                "timeout_seconds": 90,
                "train": {"num_workers": 2, "persistent_workers": True},
                "eval": {
                    "num_workers": 1,
                    "persistent_workers": False,
                    "timeout_seconds": 30,
                },
            },
        }

        train = _loader_common(config, "train")
        evaluation = _loader_common(config, "eval")

        self.assertEqual(train["num_workers"], 2)
        self.assertTrue(train["persistent_workers"])
        self.assertEqual(train["timeout"], 90)
        self.assertEqual(evaluation["num_workers"], 1)
        self.assertFalse(evaluation["persistent_workers"])
        self.assertEqual(evaluation["timeout"], 30)

    def test_root_worker_configuration_is_ignored(self) -> None:
        config = {
            "device": {"accelerator": "cpu"},
            "dataloader": {
                "num_workers": 3,
                "persistent_workers": False,
                "prefetch_factor": 4,
            },
        }

        options = _loader_common(config, "train")

        self.assertEqual(options["num_workers"], 0)
        self.assertNotIn("persistent_workers", options)
        self.assertNotIn("prefetch_factor", options)

    def test_zero_workers_omit_multiprocessing_only_options(self) -> None:
        config = {
            "device": {"accelerator": "npu"},
            "dataloader": {"train": {"num_workers": 0}},
        }

        options = _loader_common(config, "train")

        self.assertEqual(options["num_workers"], 0)
        self.assertNotIn("multiprocessing_context", options)
        self.assertNotIn("timeout", options)
        self.assertNotIn("prefetch_factor", options)
        _validate_dataloader_config(config)

    def test_invalid_worker_limits_fail_during_config_validation(self) -> None:
        config = {
            "device": {"accelerator": "npu"},
            "dataloader": {
                "train": {"num_workers": 1},
                "eval": {"num_workers": 1, "timeout_seconds": 0},
            },
        }

        with self.assertRaisesRegex(
            ValueError, "eval.timeout_seconds must be positive"
        ):
            _validate_dataloader_config(config)


if __name__ == "__main__":
    unittest.main()
