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
    """Validate the release contract: baseline, gate spec and a real report.

    Beyond "a gate exists", this checks the gate metric names and comparison
    operators (so a gate cannot be unpassable by construction) and requires at
    least one benchmark report to exist. A config with NO benchmark is not
    release-ready and FAILS (audit P0-3) -- "PASS + UNVERIFIED" was a lie.

    This command validates the CONTRACT and the existence of a benchmark; the
    artifact-level gate is ``cls-trainer release check --run ... --checkpoint
    ...``, which binds a PASS to the exact checkpoint SHA (audit P0-4).
    """
    from game_cls.reports.benchmark import (
        check_gates,
        gate_spec_fingerprint,
        validate_gate_metrics,
    )

    problems: list[str] = []
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
        # Audit P0-3: a release-ready check with no benchmark must FAIL, not
        # print PASS with an UNVERIFIED note.
        problems.append(
            f"no benchmark report under "
            f"{benchmark_cfg.get('output_dir', 'benchmarks')}; run "
            "'cls-trainer benchmark evaluate' before checking release."
        )
    elif not problems:
        problems.extend(_verify_report_against_gates(report, gate_metrics, check_gates))
    if problems:
        print("Release gate FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    print("  release contract : OK")
    print("  release gate      : PASS")
    if report is not None:
        payload = _read_report_payload(report)
        print(f"  benchmark report  : {report}")
        checkpoint_sha = (payload or {}).get("checkpoint_sha256")
        if checkpoint_sha:
            print(f"  checkpoint sha256 : {checkpoint_sha}")
        print(
            "  note              : PASS is for the newest report; run "
            "'cls-trainer release check --run <run> --checkpoint <alias|path>' "
            "to bind a PASS to one exact checkpoint (audit P0-4)."
        )
        if (
            payload
            and payload.get("gate_spec_fingerprint")
            and gate_metrics
            and payload.get("gate_spec_fingerprint")
            != gate_spec_fingerprint(gate_metrics)
        ):
            print(
                "  note              : the report's gate contract differs "
                "from this config; re-judging below.",
                    file=sys.stderr,
                )
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


def _read_report_payload(report: Path) -> dict | None:
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


def _release_identity_mismatches(
    report: Path,
    *,
    run_id: str,
    config: dict,
    gate_metrics: dict,
) -> list[str]:
    """Compare a report's full release identity against the current one.

    Audit P0-2: matching ``checkpoint_sha256`` alone is not sufficient to bind
    a PASS. FPR, recall and selection eligibility are functions of
    model + threshold + challenge set + preprocessing, not of the weights
    alone, so two runs sharing one checkpoint but differing in config,
    challenge bundle or gate spec have genuinely different benchmark results.
    Re-judging the recorded scores (``_verify_report_against_gates``) catches a
    changed gate contract, but it re-judges measurements taken under the OTHER
    run's config -- so identity must be compared before the verdict is trusted.

    A report that predates a field cannot prove it matches, so an absent field
    is a failure rather than an implicit pass; this mirrors the existing
    treatment of reports that predate the current gate set.
    """
    from game_cls.reports.benchmark import (
        canonical_config_sha256,
        file_sha256,
        gate_spec_fingerprint,
    )

    payload = _read_report_payload(report)
    if payload is None:
        return [f"benchmark report {report} is unreadable"]
    data_cfg = config.get("data") or {}
    # Must mirror cmd_benchmark_evaluate's own fingerprint definition exactly,
    # or identical inputs would compare unequal.
    challenge_source = data_cfg.get("challenge_packed_video_index") or data_cfg.get(
        "challenge_video_index"
    )
    expected = {
        "run_id": run_id,
        "resolved_config_sha256": canonical_config_sha256(config),
        "challenge_dataset_fingerprint": (
            file_sha256(challenge_source) if challenge_source else ""
        ),
        "gate_spec_fingerprint": gate_spec_fingerprint(gate_metrics),
    }
    problems: list[str] = []
    for field, current in expected.items():
        recorded = payload.get(field)
        if not recorded:
            problems.append(
                f"report records no {field}; it predates the release identity "
                "contract and cannot be bound to this artifact. Re-run "
                "'cls-trainer benchmark evaluate'."
            )
        elif not current:
            # The report claims an identity this side cannot reproduce (e.g.
            # the challenge parquet is absent), so the PASS is unverifiable.
            # A gate that cannot verify is not a gate: refuse rather than
            # skip the comparison.
            problems.append(
                f"{field} cannot be recomputed here (report={recorded}); the "
                "release identity is unverifiable. Ensure the challenge "
                "bundle and config used for the benchmark are present."
            )
        elif str(recorded) != str(current):
            problems.append(
                f"{field} differs: report={recorded} current={current}. This "
                "PASS was earned under a different release identity."
            )
    return problems


def _resolve_checkpoint_path(run_dir: Path, name: str) -> str | None:
    """Resolve a checkpoint alias/path to the artifact file without loading it."""
    direct = Path(name)
    if direct.is_file():
        return str(direct)
    checkpoints = run_dir / "checkpoints"
    for candidate in (f"model_{name}.pth", f"checkpoint_{name}.pth"):
        path = checkpoints / candidate
        if path.is_file():
            return str(path)
    return None


def _find_report_for_checkpoint(
    output_dir: Path,
    *,
    checkpoint_sha: str,
    run_id: str,
    checkpoint_alias: str,
) -> Path | None:
    """The report whose ``checkpoint_sha256`` matches the target artifact.

    Legacy reports predate the hash column, so fall back to an exact run_id +
    alias match; a report that carries a DIFFERENT checkpoint hash is never a
    match, so a later run's PASS cannot be borrowed (audit P0-4).
    """
    if not output_dir.is_dir():
        return None
    candidates = list(output_dir.glob("*/report.json"))
    for report in candidates:
        payload = _read_report_payload(report)
        if not payload:
            continue
        if payload.get("checkpoint_sha256") == checkpoint_sha:
            return report
    # Legacy fallback: a report that predates the hash column is matched by
    # exact run_id + alias. A report that CARRIES a different checkpoint hash
    # is never a match -- a later artifact cannot borrow its PASS.
    for report in candidates:
        payload = _read_report_payload(report)
        if not payload:
            continue
        if (
            not payload.get("checkpoint_sha256")
            and payload.get("run_id") == run_id
            and payload.get("checkpoint") == checkpoint_alias
        ):
            return report
    return None


def cmd_release_check(args: argparse.Namespace) -> int:
    """Bind a release PASS to one exact checkpoint (audit P0-4).

    ``config validate --release`` can only prove the contract is well-formed
    and that SOME benchmark exists. This command resolves a specific run +
    checkpoint, hashes the checkpoint, and refuses to borrow a PASS earned by
    any other artifact -- the acceptance "#5" gate for release/export.
    """
    from game_cls.cli.common import _resolve_run_dir
    from game_cls.reports.benchmark import (
        check_gates,
        file_sha256,
        validate_gate_metrics,
    )

    run_dir = _resolve_run_dir(args.run, Path(args.runs_root or "runs"))
    if not run_dir.is_dir():
        print(f"Run not found: {run_dir}", file=sys.stderr)
        return 2
    config_source = args.config or str(run_dir / "resolved_config.json")
    if not Path(config_source).is_file():
        print(
            f"No config found for run {run_dir}: pass --config or ensure "
            f"{run_dir / 'resolved_config.json'} exists.",
            file=sys.stderr,
        )
        return 2
    from game_cls.config import load_config

    config = load_config(config_source)
    benchmark_cfg = config.get("benchmark") or {}
    gate_metrics = benchmark_cfg.get("gate_metrics") or {}
    problems = list(validate_gate_metrics(gate_metrics))
    if problems:
        for problem in problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    checkpoint_path = _resolve_checkpoint_path(run_dir, args.checkpoint)
    if checkpoint_path is None:
        print(
            f"Checkpoint '{args.checkpoint}' not found under "
            f"{run_dir / 'checkpoints'}.",
            file=sys.stderr,
        )
        return 2
    checkpoint_sha = file_sha256(checkpoint_path)
    report = _find_report_for_checkpoint(
        Path(benchmark_cfg.get("output_dir", "benchmarks")),
        checkpoint_sha=checkpoint_sha,
        run_id=run_dir.name,
        checkpoint_alias=args.checkpoint,
    )
    if report is None:
        print(
            f"Release check FAILED: no benchmark PASS is bound to checkpoint "
            f"sha256={checkpoint_sha} ({checkpoint_path}). Run "
            "'cls-trainer benchmark evaluate' against this exact checkpoint "
            "before releasing it.",
            file=sys.stderr,
        )
        return 2
    # Audit P0-2: a matching checkpoint hash is NOT a release identity. FPR,
    # recall and selection eligibility are functions of model + threshold +
    # dataset + preprocessing, so a PASS earned under a different config,
    # challenge set or gate contract must never be borrowed -- even when the
    # weights are byte-identical.
    identity = _release_identity_mismatches(
        report,
        run_id=run_dir.name,
        config=config,
        gate_metrics=gate_metrics,
    )
    if identity:
        print("Release check FAILED: release identity mismatch:", file=sys.stderr)
        for problem in identity:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "The benchmark PASS was earned by a different release identity. "
            "Re-run 'cls-trainer benchmark evaluate' under this exact "
            "config/challenge/gate contract.",
            file=sys.stderr,
        )
        return 2
    verdict = _verify_report_against_gates(report, gate_metrics, check_gates)
    if verdict:
        print("Release check FAILED:", file=sys.stderr)
        for problem in verdict:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    print("  release check    : PASS")
    print(f"  run_id           : {run_dir.name}")
    print(f"  checkpoint       : {checkpoint_path}")
    print(f"  checkpoint sha256: {checkpoint_sha}")
    print(f"  benchmark report : {report}")
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
