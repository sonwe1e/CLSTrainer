"""CLSTrainer command implementation for contract 5."""

from __future__ import annotations

import sys

from game_cls.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # The implicit-train shorthand is limited to train-specific leading
    # arguments. Root options such as --help and --version must remain
    # reachable, and command-like words must go straight to argparse.
    implicit_train_options = {
        "--config",
        "--resume",
        "--fork",
        "--runs-root",
        "--dry-run",
    }
    if argv and (argv[0] in implicit_train_options or "=" in argv[0]):
        # cls-trainer --config x.yaml k=v  =>  cls-trainer train --config ...
        argv = ["train", *argv]
    if not argv:
        parser.print_help()
        return 2
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def train_command_main() -> int:
    """Entry point for tools/train.py (train subcommand only)."""
    argv = ["train", *sys.argv[1:]]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
