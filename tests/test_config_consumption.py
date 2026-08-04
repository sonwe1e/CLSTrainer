"""CI guard: every schema-known config key must have a code consumer.

step1.md requires that no configuration field can exist without an actual
consumer ("存在但不生效" is forbidden). This test scans the source tree
and asserts that every schema leaf key appears as a string literal in the
runtime code (outside the schema/config plumbing itself). A future config
addition that forgets the consumer — or a consumer removal that orphans a
key — fails CI.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from game_cls.config_schema import describe_reference

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules that define/plumb config but do not count as consumers.
NON_CONSUMER_FILES = {
    "config_schema.py",
    "config.py",
}


def _runtime_source() -> str:
    chunks: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "game_cls").rglob("*.py")):
        if path.name in NON_CONSUMER_FILES:
            continue
        chunks.append(path.read_text(encoding="utf-8"))
    for path in sorted((REPO_ROOT / "tools").glob("*.py")):
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


class ConfigConsumptionCoverageTests(unittest.TestCase):
    def test_every_schema_key_has_a_consumer(self) -> None:
        source = _runtime_source()
        orphans: list[str] = []
        for row in describe_reference():
            leaf = row["path"].split(".")[-1]
            if not re.search(rf"[\"']{re.escape(leaf)}[\"']", source):
                orphans.append(row["path"])
        self.assertEqual(
            orphans,
            [],
            "Schema keys without any code consumer: "
            + ", ".join(orphans)
            + ". Either implement the consumer or remove the key from "
            "config_schema.SCHEMA.",
        )

    def test_shipped_configs_only_use_schema_keys(self) -> None:
        # load_config applies the strict schema; any unknown key in a
        # shipped config file raises here.
        from game_cls.config import load_config

        for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
            relative = path.relative_to(REPO_ROOT)
            try:
                load_config(path)
            except FileNotFoundError:
                self.fail(f"{relative} references a missing base/layer file")
            except Exception as exc:  # noqa: BLE001 - report file + error
                self.fail(f"{relative} failed schema validation: {exc}")


if __name__ == "__main__":
    unittest.main()
