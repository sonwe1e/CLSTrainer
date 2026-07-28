from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.config import load_config
from game_cls.engine.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-frame classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    result = run_training(config)
    print(result)


if __name__ == "__main__":
    main()

