"""Allow ``python -m game_cls.cli`` (equivalent to the console script)."""

from __future__ import annotations

import sys

from game_cls.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
