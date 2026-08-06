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

from game_cls.cli.common import DEFAULT_RUNS_ROOT
from game_cls.cli.config_tools import (
    cmd_config_reference,
    cmd_config_show,
    cmd_config_validate,
)
from game_cls.cli.dataset import (
    cmd_dataset_annotate,
    cmd_dataset_audit,
    cmd_dataset_pack,
    cmd_dataset_prepare,
)
from game_cls.cli.doctor import cmd_doctor
from game_cls.cli.evaluate import cmd_evaluate
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
    validate.add_argument("overrides", nargs="*", metavar="key=value")
    validate.set_defaults(func=cmd_config_validate)
    reference = config_sub.add_parser(
        "reference", help="List every known config key with its meaning."
    )
    reference.set_defaults(func=cmd_config_reference)

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
    annotate.add_argument(
        "--metadata",
        required=True,
        help="CSV/parquet of per-video rows: source_video_uid, "
        "negative_subtype, sample_weight, ...",
    )
    annotate.add_argument(
        "--out",
        default=None,
        help="Output sidecar parquet path (default: data.metadata_sidecar).",
    )
    annotate.add_argument("overrides", nargs="*", metavar="key=value")
    annotate.set_defaults(func=cmd_dataset_annotate)

    return parser
