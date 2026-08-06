"""Re-export shim: the former monolithic module now lives in the `game_cls.engine.training` package.

Kept for import compatibility; all symbols are re-exported verbatim."""

from __future__ import annotations

from game_cls.engine.training.config_validation import (
    _dataloader_option,
    _validate_dataloader_config,
    has_independent_test,
    validate_training_config,
)  # noqa: F401
from game_cls.engine.training.early_stopping import (
    _early_stopping_defaults,
    _update_early_stopping,
)  # noqa: F401
from game_cls.engine.training.evaluation import (
    _new_interval_accumulator,
    _reduce_interval_accumulator,
    _run_evaluation,
    _threshold_weight_for_eval,
)  # noqa: F401
from game_cls.engine.training.loaders import (
    LoaderBundle,
    _build_real_data_components,
    _loader_common,
    _make_dataloaders,
    build_eval_loader_for_split,
)  # noqa: F401
from game_cls.engine.training.loop import run_training  # noqa: F401
from game_cls.engine.training.loop_util import (
    _initialize_data_worker,
    _seed_everything,
    _synchronize_device_for_metrics,
    _tb_write_scalars,
)  # noqa: F401
from game_cls.engine.training.optimizer import (
    _set_train_mode,
    build_optimizer_parameter_groups,
)  # noqa: F401
from game_cls.engine.training.run_io import (
    _EVALUATION_ROLES,
    _EVALUATION_SCOPES,
    _append_evaluation_history,
    _append_jsonl,
    _append_training_metrics,
    _evaluation_history_record,
    _finalize_run_success,
    _iso_now,
    _maybe_save_topk,
    _read_status,
    _record_run_failure,
    _write_resolved_config,
    _write_run_manifest,
)  # noqa: F401
from game_cls.engine.training.selection import (
    _LEGACY_METRIC_ALIASES,
    _annotate_selection,
    _is_better_model,
    _metric_value,
    _save_best_enabled,
    _selection_eligible,
    _selection_mode,
    _selection_rank_key,
    _selection_score,
)  # noqa: F401
from game_cls.engine.training.state import (
    _broadcast_object,
    _build_scheduler,
    _distributed_sum_int,
    _gather_random_states,
    _normalized_position,
    _save_all_ranks,
)  # noqa: F401
from game_cls.engine.training.synthetic import SyntheticPairDataset  # noqa: F401

__all__ = [
    "_read_status",
    "_iso_now",
    "_write_run_manifest",
    "_write_resolved_config",
    "_record_run_failure",
    "_finalize_run_success",
    "_append_jsonl",
    "_append_training_metrics",
    "_append_evaluation_history",
    "_evaluation_history_record",
    "_maybe_save_topk",
    "_EVALUATION_ROLES",
    "_EVALUATION_SCOPES",
    "_LEGACY_METRIC_ALIASES",
    "_metric_value",
    "_selection_mode",
    "_selection_score",
    "_selection_eligible",
    "_selection_rank_key",
    "_annotate_selection",
    "_is_better_model",
    "_save_best_enabled",
    "_threshold_weight_for_eval",
    "_new_interval_accumulator",
    "_reduce_interval_accumulator",
    "_run_evaluation",
    "_dataloader_option",
    "_validate_dataloader_config",
    "validate_training_config",
    "has_independent_test",
    "_set_train_mode",
    "build_optimizer_parameter_groups",
    "LoaderBundle",
    "_loader_common",
    "_build_real_data_components",
    "build_eval_loader_for_split",
    "_make_dataloaders",
    "_distributed_sum_int",
    "_build_scheduler",
    "_broadcast_object",
    "_gather_random_states",
    "_normalized_position",
    "_save_all_ranks",
    "_early_stopping_defaults",
    "_update_early_stopping",
    "_tb_write_scalars",
    "_seed_everything",
    "_synchronize_device_for_metrics",
    "_initialize_data_worker",
    "SyntheticPairDataset",
    "run_training",
]
