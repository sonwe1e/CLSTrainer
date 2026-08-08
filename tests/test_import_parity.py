"""Removed compatibility modules must stay unavailable in contract 5."""

from __future__ import annotations

import importlib.util
import unittest


class ImportContractTests(unittest.TestCase):
    def test_removed_trainer_module_is_unavailable(self) -> None:
        self.assertIsNone(importlib.util.find_spec("game_cls.engine.trainer"))

    def test_removed_distributed_module_is_unavailable(self) -> None:
        self.assertIsNone(importlib.util.find_spec("game_cls.engine.distributed"))


if __name__ == "__main__":
    unittest.main()
