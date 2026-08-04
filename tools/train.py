"""Training entry point.

Thin wrapper around the cls-trainer CLI ``train`` command so existing
script invocations keep working:

    python tools/train.py --config configs/cuda_debug.yaml key=value ...

Note: runs are unique by default now — each start allocates a fresh
timestamped directory under ``experiment.output_dir``. Pass
``--run-mode fixed`` to write into ``output_dir`` in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.cli import train_command_main


def main() -> None:
    sys.exit(train_command_main())


if __name__ == "__main__":
    main()
