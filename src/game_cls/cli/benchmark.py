"""cls-trainer benchmark subcommands (step5 P3/P6).

``scan-negatives`` runs a checkpoint over a training-side negative pool and
writes the versioned hard-negative mining manifest. ``evaluate`` runs a
checkpoint over the fixed challenge set, applies ``benchmark.gate_metrics``,
and writes a benchmark report. Neither touches the train/val/test protocol
nor feeds model selection.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from game_cls.cli.common import _resolve_run_dir
from game_cls.cli.evaluate import _resolve_checkpoint_state


def _reject_multi_process(command: str) -> str | None:
    """Return an error message when this command was launched under torchrun.

    Both benchmark subcommands score the whole pool on one process and write a
    single report/manifest. Under ``torchrun --nproc_per_node=N`` every rank
    would redo the identical full scan and then race on the same output path,
    so refuse instead of producing a corrupted file.
    """
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return None
    rank = os.environ.get("RANK", "?")
    return (
        f"benchmark {command} is single-process only, but WORLD_SIZE="
        f"{world_size} (RANK={rank}). Every rank would rescan the entire pool "
        "and race on the same output file. Run it without torchrun."
    )


def cmd_benchmark_scan_negatives(args: argparse.Namespace) -> int:
    """Scan a training-side negative pool and write the mining manifest."""
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.engine.checkpoint import unwrap_model
    from game_cls.engine.distributed import (
        cleanup_distributed,
        initialize_runtime,
    )
    from game_cls.engine.training.loaders import build_external_pool_loader
    from game_cls.model.builder import build_model
    from game_cls.reports.benchmark import (
        scan_negative_pool,
        write_mining_manifest,
    )

    run_dir = _resolve_run_dir(args.run, Path(args.runs_root))
    config_source = args.config or str(run_dir / "resolved_config.json")
    if not Path(config_source).is_file():
        print(
            f"No config found for run {run_dir}: pass --config or ensure "
            f"{run_dir / 'resolved_config.json'} exists.",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_config(config_source)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    guard = _reject_multi_process("scan-negatives")
    if guard:
        print(guard, file=sys.stderr)
        return 2
    mining = dict(config["data"].get("mining") or {})
    # CLI overrides win over the resolved config, then flow through the shared
    # constructor so the packed/sidecar handling applies to them too.
    if args.pool_index:
        mining["pool_index"] = args.pool_index
    if args.pool_video_index:
        mining["pool_video_index"] = args.pool_video_index
    config["data"]["mining"] = mining
    pool_index = mining.get("pool_index")
    pool_video_index = mining.get("pool_video_index")
    pool_packed_index = mining.get("pool_packed_index")
    pool_packed_video_index = mining.get("pool_packed_video_index")
    pool_metadata = mining.get("pool_metadata")
    mining_version = int(mining.get("version", 1))
    mining_enabled = bool(mining.get("enabled", False))
    has_plain_pool = bool(pool_index and pool_video_index)
    has_packed_pool = bool(pool_packed_index and pool_packed_video_index)
    if not has_plain_pool and not has_packed_pool:
        print(
            "scan-negatives needs data.mining.pool_index and "
            "data.mining.pool_video_index (plain PNG), or "
            "data.mining.pool_packed_index and "
            "data.mining.pool_packed_video_index (packed uint8).",
            file=sys.stderr,
        )
        return 2
    import logging

    logging.getLogger(__name__).info(
        "benchmark: forcing single-process mode (distributed.enabled overridden)"
    )
    config = dict(config)
    config["distributed"] = dict(config.get("distributed") or {})
    config["distributed"]["enabled"] = False
    try:
        rank, world_size, local_rank, device = initialize_runtime(config)
    except RuntimeError as exc:
        print(f"Runtime initialization failed: {exc}", file=sys.stderr)
        return 1
    del local_rank, world_size
    try:
        state_dict, checkpoint_path = _resolve_checkpoint_state(
            run_dir, args.checkpoint
        )
        model = build_model(config["model"])
        unwrap_model(model).load_state_dict(state_dict, strict=True)
        model.to(device)
        if not mining_enabled and rank == 0:
            print(
                "note: data.mining.enabled=false; running the scan explicitly.",
                file=sys.stderr,
            )
        loader, _ = build_external_pool_loader(config, pool="mining")
        rows = scan_negative_pool(
            unwrap_model(model),
            loader,
            device,
            top_k_per_video=int(mining.get("top_k_per_video", 8)),
            score_threshold=mining.get("score_threshold"),
            max_samples=mining.get("max_samples"),
        )
        out_path = Path(
            args.output or mining.get("output", "indexes/hard_negatives.parquet")
        )
        write_mining_manifest(rows, out_path)
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint_path,
                    "mined_negatives": len(rows),
                    "output": str(out_path),
                    "mining_version": mining_version,
                    "pool_metadata": pool_metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        cleanup_distributed()


def cmd_benchmark_data(args: argparse.Namespace) -> int:
    """Probe DataLoader throughput across backends/workers/prefetch."""
    import json as json_module

    from game_cls.benchmark.data_probe import run_data_probe
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    if config["data"].get("synthetic"):
        print(
            "benchmark data measures real I/O; the config uses synthetic "
            "data. Point --config at a run config with real train indexes.",
            file=sys.stderr,
        )
        return 2
    results = run_data_probe(config, steps=args.steps, batch_size=args.batch_size)
    if not results:
        print(
            "No probe variant produced data (missing indexes?). Provide a "
            "config with real train indexes or synthetic data.",
            file=sys.stderr,
        )
        return 2
    out_dir = Path(config["benchmark"].get("output_dir", "benchmarks"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_probe.json"
    out_path.write_text(
        json_module.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json_module.dumps(results, ensure_ascii=False, indent=2))
    print(f"report: {out_path}")
    return 0


def cmd_benchmark_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a checkpoint on the fixed challenge set and check gates."""
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.engine.checkpoint import unwrap_model
    from game_cls.engine.distributed import (
        cleanup_distributed,
        distributed_barrier,
        initialize_runtime,
    )
    from game_cls.engine.evaluator import evaluate
    from game_cls.engine.training.loaders import build_external_pool_loader
    from game_cls.model.builder import build_model
    from game_cls.reports.benchmark import (
        check_gates,
        validate_gate_metrics,
        write_benchmark_report,
    )

    run_dir = _resolve_run_dir(args.run, Path(args.runs_root))
    config_source = args.config or str(run_dir / "resolved_config.json")
    if not Path(config_source).is_file():
        print(
            f"No config found for run {run_dir}: pass --config or ensure "
            f"{run_dir / 'resolved_config.json'} exists.",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_config(config_source)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    guard = _reject_multi_process("evaluate")
    if guard:
        print(guard, file=sys.stderr)
        return 2
    gate_metrics = config["benchmark"].get("gate_metrics") or {}
    # Check the gate contract BEFORE scoring the challenge set: a typo'd
    # metric name or an unknown operator must not cost a full evaluation pass
    # and then surface as "metric absent". load_config already validates a
    # fresh config, but an older run's resolved_config.json predates that.
    gate_problems = validate_gate_metrics(gate_metrics)
    if gate_problems:
        for problem in gate_problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    challenge_index = config["data"].get("challenge_index")
    challenge_video_index = config["data"].get("challenge_video_index")
    challenge_packed_index = config["data"].get("challenge_packed_index")
    challenge_packed_video_index = config["data"].get("challenge_packed_video_index")
    challenge_metadata = config["data"].get("challenge_metadata")
    has_plain_challenge = bool(challenge_index and challenge_video_index)
    has_packed_challenge = bool(challenge_packed_index and challenge_packed_video_index)
    if not has_plain_challenge and not has_packed_challenge:
        print(
            "benchmark evaluate needs data.challenge_index and "
            "data.challenge_video_index (plain PNG), or "
            "data.challenge_packed_index and "
            "data.challenge_packed_video_index (packed uint8).",
            file=sys.stderr,
        )
        return 2
    import logging

    logging.getLogger(__name__).info(
        "benchmark: forcing single-process mode (distributed.enabled overridden)"
    )
    config = dict(config)
    config["distributed"] = dict(config.get("distributed") or {})
    config["distributed"]["enabled"] = False
    try:
        rank, world_size, local_rank, device = initialize_runtime(config)
    except RuntimeError as exc:
        print(f"Runtime initialization failed: {exc}", file=sys.stderr)
        return 1
    del local_rank
    try:
        state_dict, checkpoint_path = _resolve_checkpoint_state(
            run_dir, args.checkpoint
        )
        model = build_model(config["model"])
        unwrap_model(model).load_state_dict(state_dict, strict=True)
        model.to(device)
        loader, _ = build_external_pool_loader(config, pool="challenge")
        evaluation_cfg = config["evaluation"]
        result = evaluate(
            unwrap_model(model),
            loader,
            device,
            config["decision"]["threshold"],
            checkpoint_step=0,
            distributed=False,
            rank=0,
            world_size=1,
            evaluation_kind="challenge",
            report_dir=None,
            full_auc_mode=evaluation_cfg.get("full_auc_mode", "histogram"),
            auc_histogram_bins=int(evaluation_cfg.get("auc_histogram_bins", 4096)),
            amp=bool(evaluation_cfg.get("amp", config["device"].get("amp", False))),
            amp_dtype=str(
                evaluation_cfg.get(
                    "amp_dtype", config["device"].get("amp_dtype", "bfloat16")
                )
            ),
            parquet_row_group_size=int(
                evaluation_cfg.get("parquet_row_group_size", 4096)
            ),
            group_catalogs=getattr(
                getattr(loader, "dataset", None), "group_catalogs", None
            ),
            cross_entropy_weight=float(config["loss"].get("cross_entropy_weight", 1.0)),
            threshold_safety_margin=float(
                config["loss"].get("threshold_safety_margin", 0.20)
            ),
            threshold_temperature=float(
                config["loss"].get("threshold_temperature", 0.50)
            ),
            max_fpr_for_recall=float(evaluation_cfg.get("max_fpr_for_recall", 0.01)),
            tail_calibration_enabled=bool(
                evaluation_cfg.get("tail_calibration_enabled", True)
            ),
        )
        distributed_barrier()
        metrics = dict(result.metrics or {})
        gates = check_gates(metrics, gate_metrics)
        # Audit P0-4: bind the gate verdict to the exact artifact it was earned
        # on, so a later run's checkpoint can never borrow this PASS. The
        # release identity tuple is (run_id, checkpoint_sha256,
        # resolved_config_sha256, challenge_dataset_fingerprint,
        # gate_spec_fingerprint).
        from game_cls.reports.benchmark import (
            file_sha256,
            gate_spec_fingerprint,
        )

        checkpoint_sha256 = file_sha256(checkpoint_path)
        # P0-3: hash the finalized in-memory config, not the on-disk file.
        # When --config points to a different YAML than the training run,
        # file_sha256(run_dir/"resolved_config.json") would silently record
        # the wrong provenance.
        resolved_config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()
        challenge_dataset_fingerprint = file_sha256(
            challenge_packed_video_index or challenge_video_index
        )
        gate_spec_hash = gate_spec_fingerprint(gate_metrics)
        # Persist the gate verdict to run_dir so export and CI can read it
        # without re-running evaluation.
        all_passed = all(passed for _, passed, _ in gates)
        gate_report = {
            "passed": all_passed,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "run_id": run_dir.name,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "challenge_dataset_fingerprint": challenge_dataset_fingerprint,
            "gate_spec_fingerprint": gate_spec_hash,
            "gate_metrics": gate_metrics,
            "actual_metrics": {
                name: metrics.get(str(name)) for name in gate_metrics
            },
            "violations": [
                {"metric": name, "detail": detail}
                for name, passed, detail in gates
                if not passed
            ],
        }
        gate_report_path = run_dir / "benchmark_gate.json"
        gate_report_path.write_text(
            json.dumps(gate_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"gate: {'PASSED' if all_passed else 'FAILED'} — {gate_report_path}",
            file=sys.stderr,
        )
        report_path = write_benchmark_report(
            Path(config["benchmark"].get("output_dir", "benchmarks")),
            run_id=run_dir.name,
            checkpoint_alias=args.checkpoint,
            metrics=metrics,
            gates=gates,
            grouped_metrics=result.grouped_metrics,
            gate_metrics=gate_metrics,
            checkpoint_sha256=checkpoint_sha256,
            resolved_config_sha256=resolved_config_sha256,
            challenge_dataset_fingerprint=challenge_dataset_fingerprint,
            gate_spec_fingerprint=gate_spec_hash,
        )
        print(f"challenge_metadata: {challenge_metadata}", file=sys.stderr)
        unmet = [name for name, passed, _ in gates if not passed]
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "scores": {
                        "global_fpr": metrics.get("global_fpr_at_decision_threshold"),
                        "global_recall": metrics.get(
                            "global_positive_recall_at_decision_threshold"
                        ),
                        "worst_game_fpr": metrics.get(
                            "worst_game_fpr_at_decision_threshold"
                        ),
                        "worst_subtype_fpr": metrics.get(
                            "worst_subtype_fpr_at_decision_threshold"
                        ),
                    },
                    "gates": [name for name, passed, _ in gates if not passed],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1 if unmet else 0
    finally:
        cleanup_distributed()


def cmd_benchmark_gate_check(args: argparse.Namespace) -> int:
    """Check a persisted gate report without re-running evaluation.

    Exits 0 when the stored gate passed, 1 when it failed, 2 when the
    file is missing or malformed.  Lets CI pipeline scripts gate a
    deployment on a previously-measured benchmark result without paying
    the cost of a full evaluation pass.
    """
    run_dir = _resolve_run_dir(args.run, Path(args.runs_root))
    gate_report_path = run_dir / "benchmark_gate.json"
    if not gate_report_path.is_file():
        print(
            f"No gate report found at {gate_report_path}. "
            "Run 'cls-trainer benchmark evaluate' first.",
            file=sys.stderr,
        )
        return 2
    try:
        gate_data = json.loads(gate_report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Failed to read gate report {gate_report_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    passed = bool(gate_data.get("passed", False))
    print(
        json.dumps(
            {
                "passed": passed,
                "run_id": gate_data.get("run_id"),
                "checkpoint": gate_data.get("checkpoint"),
                "timestamp": gate_data.get("timestamp"),
                "violations": gate_data.get("violations", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1
