from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game_cls.config import load_and_validate_config, legacy_runtime_view
from game_cls.engine.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-frame classifier")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # Production entry point: load → migrate → validate → cross-validate.
    config = load_and_validate_config(args.config, args.overrides)

    # The legacy training loop still reads the flat V1-style config shape.
    # Convert the validated V2 config into a backwards-compatible view.
    legacy_config = legacy_runtime_view(config.model_dump())
    result = run_training(legacy_config)
    print(result)


if __name__ == "__main__":
    main()
