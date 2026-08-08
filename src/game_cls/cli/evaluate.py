"""CLSTrainer command implementation for contract 5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from game_cls.cli.common import _resolve_run_dir
from game_cls.contract import stamp_payload


def _resolve_checkpoint_state(run_dir: Path, name: str):
    """Load a model state dict from a run's checkpoints directory.

    Evaluation only needs tensors, so all loads use ``weights_only=True``;
    internal training checkpoints are recognized by their marker and their
    model sub-dict is extracted.
    """
    import torch

    direct = Path(name)
    if direct.is_file():
        payload = torch.load(direct, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"], str(direct)
        return payload, str(direct)
    checkpoints = run_dir / "checkpoints"
    model_only = checkpoints / f"model_{name}.pth"
    if model_only.is_file():
        return (
            torch.load(model_only, map_location="cpu", weights_only=True),
            str(model_only),
        )
    full_state = checkpoints / f"checkpoint_{name}.pth"
    if full_state.is_file():
        payload = torch.load(full_state, map_location="cpu", weights_only=True)
        return payload["model"], str(full_state)
    available = (
        sorted(path.name for path in checkpoints.glob("model_*.pth"))
        if checkpoints.is_dir()
        else []
    )
    raise SystemExit(
        f"Checkpoint '{name}' not found under {checkpoints}. "
        f"Available: {', '.join(available) or '(none)'}"
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate one checkpoint on the held-out test (or validation) split.

    The test split is evaluated exactly once, after training, and never
    participates in model selection: that is the whole point of the
    train/validation/test protocol.
    """
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.engine.checkpoint import unwrap_model
    from game_cls.engine.evaluator import evaluate
    from game_cls.engine.training.config_validation import has_independent_test
    from game_cls.engine.training.loaders import build_eval_loader_for_split
    from game_cls.engine.training.run_io import (
        _append_evaluation_history,
        _evaluation_history_record,
    )
    from game_cls.engine.training.selection import _annotate_selection
    from game_cls.model.builder import build_model
    from game_cls.reports.error_writer import (
        prepare_evaluation_directory,
        write_evaluation_report,
    )
    from game_cls.runtime.distributed_runtime import (
        barrier as distributed_barrier,
    )
    from game_cls.runtime.distributed_runtime import (
        cleanup as cleanup_distributed,
    )
    from game_cls.runtime.distributed_runtime import (
        init_runtime as initialize_runtime,
    )
    from game_cls.runtime.distributed_runtime import (
        is_initialized as is_distributed,
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
    if config["data"].get("synthetic"):
        print(
            "evaluate requires real indexes; the run config uses synthetic data.",
            file=sys.stderr,
        )
        return 2
    if args.split == "test" and not has_independent_test(config):
        print(
            "This run has no independent test set. Configure dedicated "
            "validation and test indexes.",
            file=sys.stderr,
        )
        return 3
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
        if rank == 0:
            print(f"Loaded checkpoint: {checkpoint_path}")
        try:
            loader, components = build_eval_loader_for_split(
                config, args.split, rank, world_size
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        kind = "test_full" if args.split == "test" else "val_full"
        report_dir = run_dir / "reports" / f"{kind}_{args.checkpoint}"
        prepare_evaluation_directory(report_dir, rank)
        distributed_barrier()
        evaluation_cfg = config["evaluation"]
        loss_cfg = config["loss"]
        result = evaluate(
            unwrap_model(model),
            loader,
            device,
            config["decision"]["threshold"],
            checkpoint_step=0,
            distributed=is_distributed(),
            rank=rank,
            world_size=world_size,
            evaluation_kind=kind,
            report_dir=report_dir,
            full_auc_mode=evaluation_cfg.get("full_auc_mode", "histogram"),
            auc_histogram_bins=int(evaluation_cfg.get("auc_histogram_bins", 4096)),
            amp=bool(evaluation_cfg.get("amp", config["device"].get("amp", False))),
            amp_dtype=str(
                evaluation_cfg.get(
                    "amp_dtype",
                    config["device"].get("amp_dtype", "bfloat16"),
                )
            ),
            parquet_row_group_size=int(
                evaluation_cfg.get("parquet_row_group_size", 4096)
            ),
            group_catalogs=getattr(
                getattr(loader, "dataset", None), "group_catalogs", None
            ),
            cross_entropy_weight=float(loss_cfg.get("cross_entropy_weight", 1.0)),
            threshold_safety_margin=float(
                loss_cfg.get("threshold_safety_margin", 0.20)
            ),
            threshold_temperature=float(loss_cfg.get("threshold_temperature", 0.50)),
            max_fpr_for_recall=float(evaluation_cfg.get("max_fpr_for_recall", 0.01)),
            tail_calibration_enabled=bool(
                evaluation_cfg.get("tail_calibration_enabled", True)
            ),
        )
        distributed_barrier()
        if rank == 0:
            metrics = dict(result.metrics or {})
            metrics.update(
                {
                    "evaluation_kind": kind,
                    "evaluation_role": (
                        "test" if args.split == "test" else "validation"
                    ),
                    "evaluation_scope": "full",
                    "checkpoint": args.checkpoint,
                    "checkpoint_path": checkpoint_path,
                    "split": args.split,
                }
            )
            _annotate_selection(metrics, evaluation_cfg)
            write_evaluation_report(
                report_dir,
                metrics,
                result.grouped_metrics,
                merge_shards=True,
                lightweight=False,
                html_max_errors=int(
                    evaluation_cfg.get("html_max_errors_per_group", 200)
                ),
                preview_decoder=getattr(
                    getattr(loader, "dataset", None), "decoder", None
                ),
            )
            _append_evaluation_history(
                run_dir,
                _evaluation_history_record(
                    metrics,
                    kind=kind,
                    global_step=int(metrics.get("checkpoint_step", 0)),
                ),
            )
            summary_path = run_dir / (
                "test_evaluation.json"
                if args.split == "test"
                else "validation_evaluation.json"
            )
            summary_path.write_text(
                json.dumps(stamp_payload(metrics), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("")
            print(f"=== {args.split} evaluation finished ===")
            print(f"checkpoint      : {checkpoint_path}")
            print(f"samples         : {metrics.get('sample_count')}")
            for key in (
                "selection_score",
                "global_f1_at_decision_threshold",
                "macro_game_f1_at_decision_threshold",
                "worst_game_f1_at_decision_threshold",
                "cross_entropy",
                "brier_score",
                "ece_20_bins",
            ):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    print(f"{key:<22}: {value:.4f}")
            print(f"report          : {report_dir}")
        distributed_barrier()
        return 0
    finally:
        cleanup_distributed()
