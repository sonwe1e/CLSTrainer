"""cls-trainer benchmark subcommands (step5 P3/P6).

``scan-negatives`` runs a checkpoint over a training-side negative pool and
writes the versioned hard-negative mining manifest. ``evaluate`` runs a
checkpoint over the fixed challenge set, applies ``benchmark.gate_metrics``,
and writes a benchmark report. Neither touches the train/val/test protocol
nor feeds model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from game_cls.cli.common import _resolve_run_dir
from game_cls.cli.evaluate import _resolve_checkpoint_state


def _build_pool_loader(config, index: str, video_index: str, batch_size: int):
    from torch.utils.data import DataLoader

    from game_cls.data.collate import pair_collate
    from game_cls.data.lazy_pair_dataset import build_eval_dataset
    from game_cls.data.video_index import read_video_entries_parquet

    test_delta = int(config["pair"]["test_delta"])
    videos = read_video_entries_parquet(video_index, (test_delta,))
    dataset: Any = build_eval_dataset(videos, test_delta, rank=0, world_size=1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=pair_collate,
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
    mining = config["data"].get("mining") or {}
    pool_index = mining.get("pool_index") or args.pool_index
    pool_video_index = mining.get("pool_video_index") or args.pool_video_index
    pool_metadata = mining.get("pool_metadata")
    mining_version = int(mining.get("version", 1))
    mining_enabled = bool(mining.get("enabled", False))
    if not pool_index or not pool_video_index:
        print(
            "scan-negatives needs data.mining.pool_index and "
            "data.mining.pool_video_index (or --pool-index/--pool-video-index).",
            file=sys.stderr,
        )
        return 2
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
        loader = _build_pool_loader(
            config,
            pool_index,
            pool_video_index,
            int(config["train"]["local_batch_size"]),
        )
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
    from game_cls.model.builder import build_model
    from game_cls.reports.benchmark import check_gates, write_benchmark_report

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
    challenge_index = config["data"].get("challenge_index")
    challenge_video_index = config["data"].get("challenge_video_index")
    challenge_metadata = config["data"].get("challenge_metadata")
    if not challenge_index or not challenge_video_index:
        print(
            "benchmark evaluate needs data.challenge_index and "
            "data.challenge_video_index (the fixed challenge set).",
            file=sys.stderr,
        )
        return 2
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
        loader = _build_pool_loader(
            config,
            challenge_index,
            challenge_video_index,
            int(config["train"]["local_batch_size"]),
        )
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
        gates = check_gates(metrics, config["benchmark"].get("gate_metrics") or {})
        report_path = write_benchmark_report(
            Path(config["benchmark"].get("output_dir", "benchmarks")),
            run_id=run_dir.name,
            checkpoint_alias=args.checkpoint,
            metrics=metrics,
            gates=gates,
            grouped_metrics=result.grouped_metrics,
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
