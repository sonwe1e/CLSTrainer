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
import sys
from pathlib import Path

from game_cls.cli.common import (
    _find_index_records,
    _flatten_dict,
    _read_run_json,
    _resolve_run_dir,
)


def cmd_run_list(args: argparse.Namespace) -> int:
    root = Path(args.root)
    records = _find_index_records(root)
    if not records:
        print(f"No runs recorded under {root.resolve()} yet.")
        return 0
    header = f"{'RUN ID':42s} {'STATE':10s} {'STEPS':>10s} FINISHED"
    print(header)
    print("-" * len(header))
    for record in records[-args.limit :]:
        print(
            f"{str(record.get('run_id') or '?'):42s} "
            f"{str(record.get('state') or '?'):10s} "
            f"{str(record.get('global_step') or '-'):>10s} "
            f"{record.get('finished', '?')}"
        )
        print(f"    {record.get('output_dir')}")
    return 0


def cmd_run_show(args: argparse.Namespace) -> int:
    root = Path(args.root)
    run_dir = _resolve_run_dir(args.target, root)
    print(f"run directory: {run_dir}")

    manifest = _read_run_json(run_dir, "manifest.json")
    status = _read_run_json(run_dir, "status.json")
    summary = _read_run_json(run_dir, "training_summary.json")
    if manifest:
        print(f"name         : {manifest.get('run_name')}")
        print(f"run id       : {manifest.get('run_id')}")
        print(f"created      : {manifest.get('created')}")
        print(f"command      : {manifest.get('command')}")
        environment = manifest.get("environment") or {}
        git = environment.get("git") or {}
        print(f"git          : {git.get('commit')} (dirty={git.get('dirty')})")
        print(f"host         : {environment.get('hostname')}")
        print(
            f"accelerator  : {manifest.get('accelerator')} "
            f"world_size={manifest.get('world_size')}"
        )
        if manifest.get("parent_run_id"):
            print(
                f"parent run   : {manifest.get('parent_run_id')} "
                f"(forked from {manifest.get('forked_from')})"
            )
    if status:
        print(f"state        : {status.get('state')}")
        print(f"last update  : {status.get('last_update')}")
        if status.get("error_type"):
            print(
                f"error        : {status['error_type']}: {status.get('error_message')}"
            )
    if summary:
        best = summary.get("best_observed_dev_test_metrics") or {}
        if isinstance(best.get("selection_score"), (int, float)):
            print(f"best score   : {best['selection_score']:.4f}")
        topk = summary.get("topk_checkpoints") or []
        if topk:
            print("topk         :")
            for entry in topk:
                value = entry.get("value")
                value_text = (
                    f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
                )
                print(
                    f"  step={entry.get('step', '?')} "
                    f"{entry.get('monitor', 'selection_score')}={value_text} "
                    f"model_{entry.get('tag', '')}.pth"
                )
    for artifact in (
        "summary.md",
        "overview.html",
        "train_metrics.jsonl",
        "console.log",
    ):
        if (run_dir / artifact).is_file():
            print(f"artifact     : {run_dir / artifact}")
    return 0


# ---------------------------------------------------------------------------
# run compare / export-tensorboard
# ---------------------------------------------------------------------------


def cmd_run_compare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    dir_a = _resolve_run_dir(args.run_a, root)
    dir_b = _resolve_run_dir(args.run_b, root)
    config_a = _read_run_json(dir_a, "resolved_config.json") or {}
    config_b = _read_run_json(dir_b, "resolved_config.json") or {}
    status_a = _read_run_json(dir_a, "status.json") or {}
    status_b = _read_run_json(dir_b, "status.json") or {}
    summary_a = _read_run_json(dir_a, "training_summary.json") or {}
    summary_b = _read_run_json(dir_b, "training_summary.json") or {}

    def headline(label: str, run_dir: Path, status: dict, summary: dict) -> None:
        best = summary.get("best_observed_dev_test_metrics") or {}
        score = best.get("selection_score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(
            f"{label}: {run_dir}\n"
            f"    state={status.get('state', '?')} "
            f"step={status.get('step', '?')} "
            f"best_selection={score_text}"
        )

    headline("A", dir_a, status_a, summary_a)
    headline("B", dir_b, status_b, summary_b)

    flat_a = _flatten_dict(config_a)
    flat_b = _flatten_dict(config_b)
    changed = sorted(
        key for key in set(flat_a) & set(flat_b) if flat_a[key] != flat_b[key]
    )
    only_a = sorted(set(flat_a) - set(flat_b))
    only_b = sorted(set(flat_b) - set(flat_a))

    print("")
    print(f"Config differences ({len(changed)} changed):")
    if not changed and not only_a and not only_b:
        print("  (configs are identical)")
    for key in changed:
        print(f"  {key}: {flat_a[key]!r} -> {flat_b[key]!r}")
    for key in only_a:
        print(f"  {key}: only in A ({flat_a[key]!r})")
    for key in only_b:
        print(f"  {key}: only in B ({flat_b[key]!r})")

    def metric_row(name: str, summary: dict) -> str:
        metrics = summary.get("best_observed_dev_test_metrics") or {}
        values = []
        for metric in (
            "global_f1_at_decision_threshold",
            "macro_game_f1_at_decision_threshold",
            "worst_game_f1_at_decision_threshold",
        ):
            value = metrics.get(
                metric,
                metrics.get(metric.replace("_at_decision_threshold", "_tau099")),
            )
            values.append(
                f"{metric}={value:.4f}"
                if isinstance(value, (int, float))
                else f"{metric}=n/a"
            )
        return f"  {name:<4s} " + " ".join(values)

    print("")
    print("Best observed dev-test metrics:")
    print(metric_row("A", summary_a))
    print(metric_row("B", summary_b))
    return 0


def cmd_run_export_tensorboard(args: argparse.Namespace) -> int:
    from game_cls.runs import read_training_metrics

    run_dir = _resolve_run_dir(args.target, Path(args.root))
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        print(
            "TensorBoard export requires the tensorboard package "
            f"(torch.utils.tensorboard unavailable: {exc}).",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out) if args.out else run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(out_dir))
    train_fields = (
        "loss",
        "ce",
        "threshold_loss",
        "interval_loss",
        "interval_ce",
        "interval_threshold_loss",
        "interval_threshold_weight",
        "interval_accuracy",
        "interval_positive_recall_tau099",
        "interval_negative_specificity_tau099",
        "interval_samples_per_second",
        "interval_step_time",
        "data_wait_ratio",
        "learning_rate",
        "grad_norm",
    )
    rows = read_training_metrics(run_dir)
    for row in rows:
        step = int(row.get("step", 0))
        for field in train_fields:
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"train/{field}", float(value), step)
    # Unified evaluation history (train probe / validation / test).
    from game_cls.runs import read_evaluation_history

    evaluation_scalars = 0
    for record in read_evaluation_history(run_dir):
        step = int(record.get("step", 0))
        split = record.get("split", "validation")
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"eval_{split}/{key}", float(value), step)
                evaluation_scalars += 1
    reports_dir = run_dir / "reports"
    evaluation_scalars = 0
    if reports_dir.is_dir():
        for metrics_file in sorted(reports_dir.glob("*/metrics.json")):
            payload = _read_run_json(metrics_file.parent, "metrics.json")
            if not payload:
                continue
            step = int(payload.get("checkpoint_step", 0))
            kind = payload.get("evaluation_kind", metrics_file.parent.name)
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(f"eval_{kind}/{key}", float(value), step)
                    evaluation_scalars += 1
    writer.close()
    print(
        f"TensorBoard events written to {out_dir} "
        f"({len(rows)} train rows, {evaluation_scalars} eval scalars)."
    )
    print(f"View with: tensorboard --logdir {out_dir}")
    return 0
