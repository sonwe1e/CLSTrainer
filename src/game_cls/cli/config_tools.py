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

import argparse
import json
import sys
from pathlib import Path
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
    problems: list[str] = []
    # Filesystem half of the hard-negative contract: semantic_validate (which
    # load_config just ran) cannot touch the filesystem, so a config pointing
    # at a non-existent sidecar reaches this point structurally valid while
    # training would silently fall back to plain negative sampling.
    from game_cls.data.sidecar import check_hard_negative_readiness

    problems.extend(check_hard_negative_readiness(config))
    if problems:
        print("Config INVALID:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    if getattr(args, "release", False):
        return _release_gate(config)
    return 0


def _release_gate(config: dict) -> int:
    """Validate the release contract: baseline, gate spec and real report.

    Beyond "a gate exists", this checks the gate metric names and comparison
    operators (so a gate cannot be unpassable by construction) and re-judges
    the newest benchmark report under the configured gates, so a config whose
    recorded run failed its own gates cannot be called releasable.
    """
    from game_cls.reports.benchmark import check_gates, validate_gate_metrics

    problems: list[str] = []
    notes: list[str] = []
    evaluation_cfg = config["evaluation"]
    if evaluation_cfg.get("minimum_worst_game_f1") is None:
        problems.append(
            "evaluation.minimum_worst_game_f1 is null; set it once a "
            "baseline exists (the release gate refuses empty runs)."
        )
    benchmark_cfg = config.get("benchmark") or {}
    gate_metrics = benchmark_cfg.get("gate_metrics") or {}
    if not gate_metrics:
        problems.append(
            "benchmark.gate_metrics is empty; the release gate needs at "
            "least one acceptance metric."
        )
    problems.extend(validate_gate_metrics(gate_metrics))
    report = _newest_benchmark_report(
        Path(benchmark_cfg.get("output_dir", "benchmarks"))
    )
    if report is None:
        notes.append(
            "no benchmark report under "
            f"{benchmark_cfg.get('output_dir', 'benchmarks')}; gates are "
            "UNVERIFIED. Run 'cls-trainer benchmark evaluate' before release."
        )
    elif not problems:
        problems.extend(_verify_report_against_gates(report, gate_metrics, check_gates))
    if problems:
        print("Release gate FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    print("  release gate      : PASS")
    for note in notes:
        print(f"  release gate note : {note}")
    if report is not None:
        print(f"  benchmark report  : {report}")
    return 0


def _newest_benchmark_report(output_dir: Path) -> Path | None:
    """Most recently written ``<run>_<alias>/report.json``, if any."""
    if not output_dir.is_dir():
        return None
    reports = sorted(
        output_dir.glob("*/report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def _verify_report_against_gates(
    report: Path, gate_metrics: dict, check_gates
) -> list[str]:
    """Re-judge a recorded benchmark report under the configured gates.

    Re-checking the recorded scores (rather than trusting the report's own
    ``passed`` flags) is what catches a report written before the gate
    contract changed, e.g. one that recorded a specificity gate as passing
    under the old substring-guessed direction.
    """
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"benchmark report {report} is unreadable: {exc}"]
    scores = payload.get("scores") or {}
    recorded = {entry.get("name") for entry in (payload.get("gates") or [])}
    problems: list[str] = []
    missing = sorted(set(map(str, gate_metrics)) - recorded)
    if missing:
        problems.append(
            f"benchmark report {report} predates the current gates and never "
            f"measured {missing}; re-run 'cls-trainer benchmark evaluate'."
        )
    measurable = {
        name: spec for name, spec in gate_metrics.items() if str(name) in scores
    }
    for name, passed, detail in check_gates(scores, measurable):
        if not passed:
            problems.append(f"benchmark report {report} fails gate {name}: {detail}")
    unmeasured = sorted(
        set(map(str, gate_metrics)) - set(map(str, measurable)) - set(missing)
    )
    if unmeasured:
        problems.append(
            f"benchmark report {report} recorded no value for {unmeasured}; "
            "the gate cannot be verified."
        )
    return problems


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
