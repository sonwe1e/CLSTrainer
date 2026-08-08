"""CLSTrainer command implementation for contract 5."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from game_cls.cli.common import (
    _read_run_json,
    _resolve_resume_checkpoint,
    _resolve_run_dir,
    _TeeContext,
    check_resume_drift,
    classify_resume,
)
from game_cls.cli.dataset import _maybe_prepare_split


def _dry_run_report(config: dict[str, Any], config_file: str) -> int:
    train_cfg = config["train"]
    evaluation_cfg = config["evaluation"]
    device_cfg = config["device"]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_batch = int(train_cfg["local_batch_size"])
    max_steps = train_cfg.get("max_steps")
    total_steps = int(
        max_steps or int(train_cfg["epochs"]) * int(train_cfg["steps_per_epoch"])
    )
    stop_after = train_cfg.get("stop_after_steps")
    checkpoint_path = config["model"].get("checkpoint_path")
    checkpoint_status = "not configured"
    if checkpoint_path:
        checkpoint_status = "exists" if Path(checkpoint_path).is_file() else "MISSING"
    data_cfg = config["data"]
    warnings: list[str] = []
    if data_cfg.get("synthetic"):
        warnings.append("data.synthetic=true — smoke-test data only.")
    if not config["model"].get("factory", "").strip():
        warnings.append("model.factory is empty.")
    if "your_package" in str(config["model"].get("factory", "")):
        warnings.append("model.factory is still the placeholder.")
    if checkpoint_path and checkpoint_status == "MISSING":
        warnings.append(f"model.checkpoint_path does not exist: {checkpoint_path}")
    from game_cls.config_schema import schedule_budget_warnings

    warnings.extend(schedule_budget_warnings(config))
    quick_every = int(evaluation_cfg.get("val_quick_every_steps", 0))
    full_every = int(evaluation_cfg.get("val_full_every_steps", 0))
    probe_every = int(evaluation_cfg.get("train_probe_every_steps", 0))

    print("=== DRY RUN — nothing will be initialized or written ===")
    print(f"config file        : {config_file}")
    print(f"accelerator        : {device_cfg.get('accelerator')}")
    print(f"world size         : {world_size} (from environment)")
    print(f"model factory      : {config['model'].get('factory')}")
    print(f"base checkpoint    : {checkpoint_path or 'none'} ({checkpoint_status})")
    print(
        "data backend       : "
        f"{data_cfg.get('backend', 'png')} "
        f"(synthetic={bool(data_cfg.get('synthetic', False))})"
    )
    print(f"decision threshold : {config['decision']['threshold']}")
    print(f"local batch size   : {local_batch}")
    print(f"global batch size  : {local_batch * world_size}")
    print(f"total steps        : {total_steps}")
    if stop_after is not None:
        print(f"stop after         : {stop_after} steps")
    print(
        f"train probe        : "
        f"{'every ' + str(probe_every) + ' steps' if probe_every else 'disabled'}"
    )
    print(
        f"quick validation   : "
        f"{'every ' + str(quick_every) + ' steps' if quick_every else 'disabled'}"
    )
    print(
        f"full validation    : "
        f"{'every ' + str(full_every) + ' steps' if full_every else 'disabled'}"
        f" (at end: {bool(evaluation_cfg.get('val_full_at_end', True))})"
    )
    data_cfg = config["data"]
    if data_cfg.get("synthetic"):
        print("split roles        : synthetic (train/val/test)")
    else:
        print("split roles        : train / validation / test")
    early_cfg = config.get("early_stopping") or {}
    if early_cfg.get("enabled"):
        print(
            f"early stopping     : monitor={early_cfg.get('monitor')} "
            f"patience={early_cfg.get('patience_evaluations')} "
            f"min_delta={early_cfg.get('min_delta')} "
            f"burn_in={early_cfg.get('burn_in_steps')}"
        )
    else:
        print("early stopping     : disabled")
    print(f"runs root          : {config['experiment']['output_dir']}")
    print(
        "planned output     : <runs root>/<date>/<time>_<name>_<id>/ "
        "(allocated at start)"
    )
    print(f"config warnings    : {len(warnings)}")
    for warning in warnings:
        print(f"  [WARNING] {warning}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.engine.training.loop import run_training
    from game_cls.runs import collect_environment, sha256_file

    try:
        if args.resume and args.fork:
            raise SystemExit("--resume and --fork are mutually exclusive.")
        if args.resume:
            resume_dir = Path(args.resume).resolve()
            if not resume_dir.is_dir():
                raise SystemExit(f"Resume run directory does not exist: {resume_dir}")
            config_source = args.config or str(resume_dir / "resolved_config.json")
            if not Path(config_source).is_file():
                raise SystemExit(
                    f"No config found for resume: pass --config or ensure "
                    f"{resume_dir / 'resolved_config.json'} exists."
                )
        elif args.fork:
            fork_dir = _resolve_run_dir(args.fork, Path(args.runs_root))
            config_source = args.config or str(fork_dir / "resolved_config.json")
            if not Path(config_source).is_file():
                raise SystemExit(
                    f"No config found for fork: pass --config or ensure "
                    f"{fork_dir / 'resolved_config.json'} exists."
                )
        else:
            if not args.config:
                raise SystemExit("train requires --config (or --resume/--fork).")
            config_source = args.config
        config = load_config(config_source, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2

    parent_run_id: str | None = None
    forked_from: str | None = None
    resume_run_id: str | None = None
    inferred_runs_root: str | None = None
    resume_type = "exact"
    resume_config_diffs: list[str] = []
    if args.resume:
        baseline = _read_run_json(resume_dir, "resolved_config.json")
        if baseline:
            critical, drift_warnings = check_resume_drift(baseline, config)
            for warning in drift_warnings:
                print(f"[WARNING] resume config drift: {warning}")
            if critical:
                print(
                    "Resume refused: critical config drift detected:",
                    file=sys.stderr,
                )
                for problem in critical:
                    print(f"  - {problem}", file=sys.stderr)
                print(
                    "These fields change what the restored state means. "
                    "Start a new run (optionally with --fork) instead.",
                    file=sys.stderr,
                )
                return 3
            resume_config_diffs = drift_warnings
            resume_type = classify_resume(baseline, config, drift_warnings)
            if resume_type == "fork":
                print(
                    "Resume refused: this would change the training "
                    "strategy, which invalidates the restored state. "
                    "Use --fork RUN_ID to start a new run derived from "
                    "this one instead.",
                    file=sys.stderr,
                )
                return 3
            if resume_type == "extend":
                print(
                    "[WARNING] resume-extend: max_steps/stop_after changed; "
                    "the scheduler is re-planned for the new budget."
                )
        resume_manifest = _read_run_json(resume_dir, "manifest.json") or {}
        resume_run_id = resume_manifest.get("run_id")
        # Unique-mode runs live at <runs_root>/<YYYYMMDD>/<run_id>; recover
        # the original runs root so index/status updates land next to the
        # run's siblings instead of inside the run directory.
        if resume_run_id:
            candidate = resume_dir.parent.parent
            if candidate.is_dir():
                inferred_runs_root = str(candidate)
        config["experiment"]["output_dir"] = str(resume_dir)
        config["train"]["resume_path"] = _resolve_resume_checkpoint(resume_dir, config)
    elif args.fork:
        fork_manifest = _read_run_json(fork_dir, "manifest.json") or {}
        parent_run_id = fork_manifest.get("run_id")
        forked_from = str(fork_dir)
        explicit_resume = any(
            override.split("=", 1)[0] == "train.resume_path"
            for override in args.overrides
        )
        if not explicit_resume:
            config["train"]["resume_path"] = None

    if args.dry_run:
        return _dry_run_report(config, config_source)

    _maybe_prepare_split(config)

    checkpoint_path = config["model"].get("checkpoint_path")
    checkpoint_hash = None
    rank = int(os.environ.get("RANK", 0))
    if checkpoint_path and Path(checkpoint_path).is_file() and rank == 0:
        if os.environ.get("CLS_SKIP_CHECKPOINT_HASH"):
            checkpoint_hash = "skipped"
        else:
            checkpoint_hash = sha256_file(checkpoint_path)

    run_meta = {
        "command": " ".join(sys.argv),
        "config_file": config_source,
        "resumed_from": args.resume,
        "forked_from": forked_from,
        "parent_run_id": parent_run_id,
        "run_id": resume_run_id,
        "resume_type": resume_type,
        "resume_config_diffs": resume_config_diffs,
        "runs_root": inferred_runs_root or str(Path(args.runs_root).resolve()),
        "base_checkpoint_sha256": checkpoint_hash,
        "environment": collect_environment(config),
    }

    with _TeeContext() as tee:
        result = run_training(config, run_meta=run_meta, on_run_dir=tee.attach)

    output_dir = result.get("output_dir", "?")
    run_dir = Path(output_dir)
    print("")
    print("=== Training finished ===")
    print(f"state        : {result.get('state')}")
    print(f"run directory: {output_dir}")
    print(f"final step   : {result.get('global_step')}")
    best = result.get("best_metrics") or {}
    if isinstance(best, dict) and best:
        score = best.get("selection_score")
        if isinstance(score, (int, float)):
            print(f"best selection score: {score:.4f}")
    print("")
    print("Where to look:")
    print(f"  human summary     : {run_dir / 'summary.md'}")
    print(f"  status            : {run_dir / 'status.json'}")
    print(f"  manifest          : {run_dir / 'manifest.json'}")
    print(f"  training curves   : {run_dir / 'train_metrics.jsonl'}")
    print(f"  evaluation reports: {run_dir / 'reports'}")
    print(f"  checkpoints       : {run_dir / 'checkpoints'}")
    return 0
