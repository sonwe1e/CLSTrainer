"""Legacy two-split config migration onto the train/val/test protocol."""

from __future__ import annotations

import unittest

from game_cls.config import load_config
from game_cls.config_schema import (
    finalize_config,
    split_role_warnings,
)


class SplitMigrationTests(unittest.TestCase):
    def test_legacy_evaluation_keys_migrate_to_val_keys(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["evaluation"].pop("val_quick_every_steps", None)
        config["evaluation"].pop("val_quick_pairs_per_video", None)
        config["evaluation"]["quick_test_every_steps"] = 7
        config["evaluation"]["quick_test_pairs_per_video"] = 9
        finalized = finalize_config(config)
        self.assertEqual(finalized["evaluation"]["val_quick_every_steps"], 7)
        self.assertEqual(finalized["evaluation"]["val_quick_pairs_per_video"], 9)
        self.assertNotIn("quick_test_every_steps", finalized["evaluation"])
        self.assertNotIn("quick_test_pairs_per_video", finalized["evaluation"])

    def test_missing_val_index_aliases_test_and_warns(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        data = config["data"]
        # load_config finalizes, so the alias is already applied.
        self.assertEqual(data["val_index"], data["test_index"])
        migration = data.get("split_migration") or {}
        self.assertTrue(migration.get("test_used_as_validation"))
        warnings = split_role_warnings(config)
        self.assertEqual(len(warnings), 1)
        self.assertIn("NO independent test set", warnings[0])

    def test_finalize_is_idempotent_after_migration(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        once = finalize_config(config)
        twice = finalize_config(once)
        self.assertEqual(once, twice)

    def test_independent_val_and_test_splits_do_not_warn(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["data"]["val_index"] = "indexes/val_frames.parquet"
        config["data"]["val_video_index"] = "indexes/val_video_entries.parquet"
        config["data"].pop("split_migration", None)
        finalized = finalize_config(config)
        self.assertFalse(
            finalized["data"]["split_migration"].get("test_used_as_validation")
        )
        self.assertEqual(split_role_warnings(finalized), [])


if __name__ == "__main__":
    unittest.main()
