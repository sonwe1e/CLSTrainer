"""cls-trainer command line interface.

Workflows:

    cls-trainer train --config configs/recipes/game_cls_production.yaml [key=value ...]
    cls-trainer train --config ... --dry-run
    cls-trainer train --resume <run_dir>
    cls-trainer config show --config ... [--with-source]
    cls-trainer config validate --config ...
    cls-trainer config reference
    cls-trainer run list [--root runs]
    cls-trainer run show latest|<run_dir>
    cls-trainer doctor --config ...

Every ``train`` start defaults to ``--run-mode unique``: the configured
``experiment.output_dir`` is treated as a runs root and a fresh timestamped
run directory is allocated, so re-running a command can never overwrite a
previous run. ``--run-mode fixed`` restores the legacy in-place behavior.
"""

from __future__ import annotations

import sys

from game_cls.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    known = {
        "train",
        "config",
        "run",
        "doctor",
        "init",
        "evaluate",
        "dataset",
        "benchmark",
        "export",
    }
    if not argv or argv[0] not in known:
        # cls-trainer --config x.yaml k=v  =>  cls-trainer train --config ...
        argv = ["train", *argv]
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def train_command_main() -> int:
    """Entry point for tools/train.py (train subcommand only)."""
    argv = ["train", *sys.argv[1:]]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
