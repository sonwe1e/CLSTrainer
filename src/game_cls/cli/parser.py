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

from game_cls.cli.benchmark import (
    cmd_benchmark_data,
    cmd_benchmark_evaluate,
    cmd_benchmark_gate_check,
    cmd_benchmark_scan_negatives,
)
from game_cls.cli.common import DEFAULT_RUNS_ROOT
from game_cls.cli.config_tools import (
    cmd_config_reference,
    cmd_config_show,
    cmd_config_validate,
    cmd_release_check,
)
from game_cls.cli.dataset import (
    cmd_dataset_annotate,
    cmd_dataset_audit,
    cmd_dataset_pack,
    cmd_dataset_prepare,
)
from game_cls.cli.doctor import cmd_doctor
from game_cls.cli.evaluate import cmd_evaluate
from game_cls.cli.export import cmd_export
from game_cls.cli.init_cmd import cmd_init
from game_cls.cli.run_tools import (
    cmd_run_compare,
    cmd_run_export_tensorboard,
    cmd_run_list,
    cmd_run_show,
)
from game_cls.cli.train import cmd_train

# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cls-trainer",
        description="Dual-frame multi-game binary classification trainer.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser(
        "train", help="Train the classifier (unique run directory by default)."
    )
    train.add_argument(
        "--config",
        help="Config file. Optional with --resume (defaults to the run's "
        "resolved_config.json).",
    )
    train.add_argument(
        "--run-mode",
        choices=("unique", "fixed"),
        default="unique",
        help=(
            "unique (default): allocate an immutable timestamped run dir "
            "under experiment.output_dir. fixed: write into output_dir in "
            "place (legacy)."
        ),
    )
    train.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="Continue an existing run directory (implies fixed mode).",
    )
    train.add_argument(
        "--fork",
        metavar="RUN",
        help="Start a new run derived from an existing run's config "
        "(records parent lineage; implies unique mode).",
    )
    train.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Runs root used to resolve --fork targets by run id.",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan; initialize nothing.",
    )
    train.add_argument("overrides", nargs="*", metavar="key=value")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on the held-out test (or validation) "
        "split; the test split never participates in model selection.",
    )
    evaluate.add_argument(
        "--run",
        required=True,
        help="Run directory, run id or 'latest'.",
    )
    evaluate.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Runs root used to resolve --run by run id.",
    )
    evaluate.add_argument(
        "--checkpoint",
        default="best_selection",
        help=(
            "Checkpoint alias (last, best_selection, best_val_loss, "
            "best_worst_game) or a direct .pth path."
        ),
    )
    evaluate.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Split to evaluate; 'test' requires an independent test set.",
    )
    evaluate.add_argument(
        "--config",
        help="Optional config override (defaults to the run's resolved_config.json).",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    config = subparsers.add_parser("config", help="Config utilities.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    show = config_sub.add_parser(
        "show", help="Print the resolved (finalized) configuration."
    )
    show.add_argument("--config", required=True)
    show.add_argument(
        "--with-source",
        action="store_true",
        help="Annotate every value with the file/override that set it.",
    )
    show.add_argument("overrides", nargs="*", metavar="key=value")
    show.set_defaults(func=cmd_config_show)
    validate = config_sub.add_parser(
        "validate", help="Load and validate a config without training."
    )
    validate.add_argument("--config", required=True)
    validate.add_argument(
        "--release",
        action="store_true",
        help="Release gate: require a non-null minimum_worst_game_f1 and a "
        "non-empty benchmark.gate_metrics.",
    )
    validate.add_argument("overrides", nargs="*", metavar="key=value")
    validate.set_defaults(func=cmd_config_validate)
    reference = config_sub.add_parser(
        "reference", help="List every known config key with its meaning."
    )
    reference.set_defaults(func=cmd_config_reference)

    release = subparsers.add_parser(
        "release", help="Bind a release PASS to one exact artifact."
    )
    release_sub = release.add_subparsers(dest="release_command", required=True)
    release_check = release_sub.add_parser(
        "check",
        help="Verify one run's checkpoint against its own benchmark gate "
        "(audit P0-4: a PASS cannot be borrowed from another artifact).",
    )
    release_check.add_argument("--run", required=True)
    release_check.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint alias (e.g. best_selection) or path to a model .pth.",
    )
    release_check.add_argument("--config")
    release_check.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    release_check.set_defaults(func=cmd_release_check)

    run = subparsers.add_parser("run", help="Inspect recorded runs.")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_list = run_sub.add_parser("list", help="List runs from the index.")
    run_list.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_list.add_argument("--limit", type=int, default=20)
    run_list.set_defaults(func=cmd_run_list)
    run_show = run_sub.add_parser(
        "show", help="Show one run ('latest' or a run directory)."
    )
    run_show.add_argument("target", help="'latest', a run directory or run id.")
    run_show.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_show.set_defaults(func=cmd_run_show)
    run_compare = run_sub.add_parser(
        "compare", help="Compare two runs: config diff + best metrics."
    )
    run_compare.add_argument("run_a", help="Run directory, run id or 'latest'.")
    run_compare.add_argument("run_b", help="Run directory, run id or 'latest'.")
    run_compare.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_compare.set_defaults(func=cmd_run_compare)
    run_export = run_sub.add_parser(
        "export-tensorboard",
        help="Export a run's curves to TensorBoard event files.",
    )
    run_export.add_argument("target", help="Run directory, run id or 'latest'.")
    run_export.add_argument("--out", help="Output logdir (default <run>/tensorboard).")
    run_export.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_export.set_defaults(func=cmd_run_export_tensorboard)

    doctor = subparsers.add_parser(
        "doctor", help="Check environment, config, data and model readiness."
    )
    doctor.add_argument("--config", required=True)
    doctor.add_argument("overrides", nargs="*", metavar="key=value")
    doctor.set_defaults(func=cmd_doctor)

    init = subparsers.add_parser(
        "init", help="Create a minimal recipe for a given profile."
    )
    init.add_argument(
        "--profile",
        default="npu_8p",
        help="Machine profile (see configs/profiles/).",
    )
    init.add_argument("--name", default="my_run", help="Run name.")
    init.add_argument(
        "--output",
        default="configs/recipes/my_run.yaml",
        help="Recipe file to create.",
    )
    init.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    init.set_defaults(func=cmd_init)

    dataset = subparsers.add_parser(
        "dataset", help="Dataset preparation, audit and packing."
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_sub.add_parser(
        "prepare",
        help="Derive a source-video-level validation split from train_root.",
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--train-root", required=True)
    prepare.add_argument("--test-root", required=True)
    prepare.add_argument("--output-dir", default="indexes")
    prepare.add_argument(
        "--val-ratio",
        type=float,
        help="Override data.split.val_ratio for this preparation.",
    )
    prepare.add_argument("overrides", nargs="*", metavar="key=value")
    prepare.set_defaults(func=cmd_dataset_prepare)
    audit = dataset_sub.add_parser("audit", help="Validate an index audit report.")
    audit.add_argument("--config", required=True)
    audit.add_argument("--index-dir", default="indexes")
    audit.add_argument(
        "--strict",
        action="store_true",
        help="Run the strict validate_audit gate (mirrors tools/audit_dataset.py).",
    )
    audit.add_argument("overrides", nargs="*", metavar="key=value")
    audit.set_defaults(func=cmd_dataset_audit)
    pack = dataset_sub.add_parser(
        "pack", help="Pack decoded CHW uint8 frames into memmapped shards."
    )
    pack.add_argument("--config", required=True)
    pack.add_argument("--frame-index", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--images-per-shard", type=int, default=4096)
    pack.add_argument("overrides", nargs="*", metavar="key=value")
    pack.set_defaults(func=cmd_dataset_pack)
    annotate = dataset_sub.add_parser(
        "annotate",
        help="Import per-video metadata into the metadata sidecar "
        "(negative_subtype, sample_weight).",
    )
    annotate.add_argument("--config", required=True)
    # Exactly one input source. --metadata used to be required=True, which
    # forced a meaningless --metadata alongside every --from-mining run.
    annotate_source = annotate.add_mutually_exclusive_group(required=True)
    annotate_source.add_argument(
        "--metadata",
        help="CSV/parquet of per-video rows: source_video_uid, "
        "negative_subtype, sample_weight, ...",
    )
    annotate_source.add_argument(
        "--from-mining",
        help="Instead of --metadata, import mined negatives from a "
        "hard_negatives.parquet manifest.",
    )
    annotate.add_argument(
        "--out",
        default=None,
        help="Output sidecar parquet path (default: data.metadata_sidecar).",
    )
    annotate.add_argument(
        "--subtype",
        default=None,
        help="negative_subtype assigned to mined videos (requires --from-mining).",
    )
    annotate.add_argument(
        "--on-subtype-conflict",
        choices=("refuse", "keep", "overwrite"),
        default="refuse",
        help="Existing sidecar row already carries a different "
        "negative_subtype: refuse the import (default), keep the existing "
        "annotation, or overwrite it.",
    )
    annotate.add_argument("overrides", nargs="*", metavar="key=value")
    annotate.set_defaults(func=cmd_dataset_annotate)

    export = subparsers.add_parser(
        "export",
        help="Export a checkpoint as a deployment artifact (weights|onnx).",
    )
    export.add_argument("--config")
    export.add_argument("--run", required=True)
    export.add_argument(
        "--checkpoint",
        default="best_selection",
        help="Checkpoint alias/path (default: best_selection).",
    )
    export.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    # --format/--out default to None on purpose: an argparse default would
    # shadow export.format / export.output_dir on every plain invocation.
    # Unset falls back to the config, then to weights/exports.
    export.add_argument(
        "--format",
        choices=("weights", "onnx"),
        default=None,
        help="Override export.format. weights: pure state dict + manifest "
        "(default). onnx: traced [B,2] graph verified against the PyTorch "
        "reference.",
    )
    export.add_argument(
        "--out",
        default=None,
        help="Override export.output_dir (default: exports).",
    )
    export.add_argument(
        "--skip-gate",
        action="store_true",
        help="Export even when this checkpoint has no benchmark PASS bound to "
        "it. Default refuses: an artifact without an exact-checkpoint gate "
        "report is not releasable (audit P0-4).",
    )
    export.set_defaults(func=cmd_export)

    benchmark = subparsers.add_parser(
        "benchmark", help="Hard-negative mining and challenge-set benchmarking."
    )
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    scan = benchmark_sub.add_parser(
        "scan-negatives",
        help="Score a negative pool with a checkpoint and write the mining manifest.",
    )
    scan.add_argument("--config")
    scan.add_argument("--run", required=True)
    scan.add_argument(
        "--checkpoint",
        default="best_selection",
        help="Checkpoint alias/path (default: best_selection).",
    )
    scan.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    scan.add_argument("--pool-index", default=None)
    scan.add_argument("--pool-video-index", default=None)
    scan.add_argument("--output", default=None)
    scan.set_defaults(func=cmd_benchmark_scan_negatives)
    bench_data = benchmark_sub.add_parser(
        "data",
        help="Probe DataLoader throughput across backends/workers/prefetch.",
    )
    bench_data.add_argument("--config", required=True)
    bench_data.add_argument("--steps", type=int, default=20)
    bench_data.add_argument("--batch-size", type=int, default=16)
    bench_data.add_argument("overrides", nargs="*", metavar="key=value")
    bench_data.set_defaults(func=cmd_benchmark_data)
    bench_eval = benchmark_sub.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on the fixed challenge set and check "
        "benchmark.gate_metrics.",
    )
    bench_eval.add_argument("--config")
    bench_eval.add_argument("--run", required=True)
    bench_eval.add_argument(
        "--checkpoint",
        default="best_selection",
        help="Checkpoint alias/path (default: best_selection).",
    )
    bench_eval.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    bench_eval.set_defaults(func=cmd_benchmark_evaluate)
    bench_gate_check = benchmark_sub.add_parser(
        "gate-check",
        help="Check a persisted benchmark_gate.json without re-running evaluation. "
        "Exits 0 (passed), 1 (failed), or 2 (file missing/malformed).",
    )
    bench_gate_check.add_argument("--run", required=True)
    bench_gate_check.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    bench_gate_check.set_defaults(func=cmd_benchmark_gate_check)

    return parser
