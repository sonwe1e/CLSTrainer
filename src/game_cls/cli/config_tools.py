"""cls-trainer command line interface.

Workflows:

    cls-trainer train --config configs/npu_1p.yaml [key=value ...]
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

import argparse
import json
import sys
from typing import Any

# ---------------------------------------------------------------------------
# config show / validate / reference
# ---------------------------------------------------------------------------


def cmd_config_show(args: argparse.Namespace) -> int:
    from game_cls.config import load_config, load_config_with_sources

    if args.with_source:
        config, sources = load_config_with_sources(args.config, args.overrides)
        from game_cls.config_schema import describe_reference

        order = {row["path"]: i for i, row in enumerate(describe_reference())}

        def flatten(node: dict, prefix: str = "") -> list[tuple[str, Any]]:
            rows: list[tuple[str, Any]] = []
            for key, value in node.items():
                dotted = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    rows.extend(flatten(value, dotted))
                else:
                    rows.append((dotted, value))
            return rows

        known = {dotted for dotted, _ in flatten(config)}
        flattened = flatten(config)
        flattened.sort(key=lambda item: order.get(item[0], 10_000))
        for dotted, value in flattened:
            origin = sources.get(dotted, "<default>")
            print(f"{dotted} = {value!r}    # from {origin}")
        extra_sources = {
            dotted: origin for dotted, origin in sources.items() if dotted not in known
        }
        for dotted, origin in sorted(extra_sources.items()):
            print(f"{dotted} = <not set>    # from {origin}")
        return 0

    config = load_config(args.config, args.overrides)
    try:
        import yaml

        print(yaml.safe_dump(config, sort_keys=True, allow_unicode=True))
    except ImportError:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        print("Config INVALID:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    train_cfg = config["train"]
    total_steps = int(
        train_cfg.get("max_steps")
        or int(train_cfg["epochs"]) * int(train_cfg["steps_per_epoch"])
    )
    print(f"Config OK: {args.config}")
    print(f"  decision threshold: {config['decision']['threshold']}")
    print(f"  total steps       : {total_steps}")
    print(f"  local batch size  : {train_cfg['local_batch_size']}")
    print(f"  output dir        : {config['experiment']['output_dir']}")
    return 0


def cmd_config_reference(args: argparse.Namespace) -> int:
    from game_cls.config_schema import describe_reference

    for row in describe_reference():
        flags = []
        if row["legacy"]:
            flags.append("legacy")
        if row["choices"]:
            flags.append("choices: " + "|".join(map(str, row["choices"])))
        suffix = f" ({'; '.join(flags)})" if flags else ""
        print(f"{row['path']}  [{row['type']}]{suffix}")
        print(f"    {row['description']}")
    return 0
