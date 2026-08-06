"""Re-export shim: the former monolithic module now lives in the `game_cls.cli` package.

Kept for import compatibility; all symbols are re-exported verbatim."""

from __future__ import annotations

from game_cls.cli.common import (
    _CHECKPOINT_ALIASES,
    _MAX_TEE_BUFFER_BYTES,
    DEFAULT_RUNS_ROOT,
    RESUME_CRITICAL_DIFFS,
    RESUME_EXPECTED_DIFFS,
    RESUME_EXTEND_KEYS,
    _find_index_records,
    _flatten_dict,
    _read_run_json,
    _resolve_resume_checkpoint,
    _resolve_run_dir,
    _StreamTee,
    _TeeContext,
    check_resume_drift,
    classify_resume,
)  # noqa: F401
from game_cls.cli.config_tools import (
    cmd_config_reference,
    cmd_config_show,
    cmd_config_validate,
)  # noqa: F401
from game_cls.cli.dataset import (
    _maybe_prepare_split,
    _run_split_prepare,
    cmd_dataset_audit,
    cmd_dataset_pack,
    cmd_dataset_prepare,
)  # noqa: F401
from game_cls.cli.doctor import cmd_doctor  # noqa: F401
from game_cls.cli.evaluate import _resolve_checkpoint_state, cmd_evaluate  # noqa: F401
from game_cls.cli.init import main, train_command_main  # noqa: F401
from game_cls.cli.init_cmd import RECIPE_TEMPLATE, cmd_init  # noqa: F401
from game_cls.cli.parser import build_parser  # noqa: F401
from game_cls.cli.run_tools import (
    cmd_run_compare,
    cmd_run_export_tensorboard,
    cmd_run_list,
    cmd_run_show,
)  # noqa: F401
from game_cls.cli.train import _dry_run_report, cmd_train  # noqa: F401

__all__ = [
    "build_parser",
    "DEFAULT_RUNS_ROOT",
    "_MAX_TEE_BUFFER_BYTES",
    "RESUME_EXPECTED_DIFFS",
    "RESUME_EXTEND_KEYS",
    "RESUME_CRITICAL_DIFFS",
    "_CHECKPOINT_ALIASES",
    "_flatten_dict",
    "check_resume_drift",
    "classify_resume",
    "_resolve_run_dir",
    "_read_run_json",
    "_resolve_resume_checkpoint",
    "_StreamTee",
    "_TeeContext",
    "_find_index_records",
    "_dry_run_report",
    "cmd_train",
    "_resolve_checkpoint_state",
    "cmd_evaluate",
    "cmd_config_show",
    "cmd_config_validate",
    "cmd_config_reference",
    "cmd_run_list",
    "cmd_run_show",
    "cmd_run_compare",
    "cmd_run_export_tensorboard",
    "cmd_doctor",
    "RECIPE_TEMPLATE",
    "cmd_init",
    "_run_split_prepare",
    "_maybe_prepare_split",
    "cmd_dataset_prepare",
    "cmd_dataset_audit",
    "cmd_dataset_pack",
    "main",
    "train_command_main",
]
