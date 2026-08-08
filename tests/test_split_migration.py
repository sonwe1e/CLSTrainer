"""Contract 5 rejects removed split/config compatibility paths."""

from __future__ import annotations

import unittest

from game_cls.config import load_config
from game_cls.config_schema import ConfigSchemaError, finalize_config


class SplitContractTests(unittest.TestCase):
    def test_removed_evaluation_key_is_rejected(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["evaluation"]["quick_test_every_steps"] = 7
        with self.assertRaises(ConfigSchemaError):
            finalize_config(config)

    def test_missing_validation_index_is_not_aliased(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["data"]["test_index"] = "indexes/test_frames.parquet"
        config["data"].pop("val_index", None)
        finalized = finalize_config(config)
        self.assertNotIn("val_index", finalized["data"])
        self.assertNotIn("split_migration", finalized["data"])

    def test_finalize_is_idempotent(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        self.assertEqual(
            finalize_config(config), finalize_config(finalize_config(config))
        )


if __name__ == "__main__":
    unittest.main()
