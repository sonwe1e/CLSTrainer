from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .distributed import cleanup_runtime, init_runtime
from .plotting import plot_history
from .trainer import check_setup, evaluate_checkpoint, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clstrainer-lite",
        description="Small dual-frame binary classification trainer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "check"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("overrides", nargs="*", metavar="key=value")

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("overrides", nargs="*", metavar="key=value")

    plot = sub.add_parser("plot")
    plot.add_argument("--history", required=True, help="Path to history.json")
    plot.add_argument("--out", help="Output directory; defaults to history file directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plot":
        history_path = Path(args.history)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        plot_history(history, Path(args.out) if args.out else history_path.parent)
        return 0

    config = load_config(args.config, args.overrides)
    runtime = init_runtime(config["runtime"])
    try:
        if args.command == "check":
            check_setup(config, runtime)
        elif args.command == "train":
            train(config, runtime)
        elif args.command == "eval":
            evaluate_checkpoint(config, runtime, args.checkpoint, split=args.split)
        else:
            raise AssertionError(args.command)
        return 0
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
