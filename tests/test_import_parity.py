"""P1 guard: the trainer.py shim re-exports the exact moved symbols.

After the monolithic ``engine/trainer.py`` was split into the
``engine/training`` package, the original module path must keep resolving to
the *same* objects as the package submodules (imports elsewhere in the repo
and in tests rely on it). If a future edit redefines a symbol in the shim or
moves a function without updating the shim, this test fails.
"""

from __future__ import annotations

import unittest

import game_cls.engine.trainer as trainer
import game_cls.engine.training.loop as loop


# Mirrors the split map; every top-level name the old trainer.py exported.
SHIM_OWNER: dict[str, str] = {
    "run_training": "loop",
    "SyntheticPairDataset": "synthetic",
    "LoaderBundle": "loaders",
    # run_io
    "_read_status": "run_io",
    "_iso_now": "run_io",
    "_write_run_manifest": "run_io",
    "_write_resolved_config": "run_io",
    "_record_run_failure": "run_io",
    "_finalize_run_success": "run_io",
    "_append_jsonl": "run_io",
    "_append_training_metrics": "run_io",
    "_append_evaluation_history": "run_io",
    "_evaluation_history_record": "run_io",
    "_maybe_save_topk": "run_io",
    "_EVALUATION_ROLES": "run_io",
    "_EVALUATION_SCOPES": "run_io",
    # selection
    "_LEGACY_METRIC_ALIASES": "selection",
    "_metric_value": "selection",
    "_selection_mode": "selection",
    "_selection_score": "selection",
    "_selection_eligible": "selection",
    "_selection_rank_key": "selection",
    "_annotate_selection": "selection",
    "_is_better_model": "selection",
    "_save_best_enabled": "selection",
    # evaluation
    "_threshold_weight_for_eval": "evaluation",
    "_new_interval_accumulator": "evaluation",
    "_reduce_interval_accumulator": "evaluation",
    "_run_evaluation": "evaluation",
    # config_validation
    "_dataloader_option": "config_validation",
    "_validate_dataloader_config": "config_validation",
    "validate_training_config": "config_validation",
    "has_independent_test": "config_validation",
    # optimizer
    "_set_train_mode": "optimizer",
    "build_optimizer_parameter_groups": "optimizer",
    # loaders
    "_loader_common": "loaders",
    "_build_real_data_components": "loaders",
    "build_eval_loader_for_split": "loaders",
    "_make_dataloaders": "loaders",
    # state
    "_distributed_sum_int": "state",
    "_build_scheduler": "state",
    "_broadcast_object": "state",
    "_gather_random_states": "state",
    "_normalized_position": "state",
    "_save_all_ranks": "state",
    # early_stopping
    "_early_stopping_defaults": "early_stopping",
    "_update_early_stopping": "early_stopping",
    # loop_util
    "_tb_write_scalars": "loop_util",
    "_seed_everything": "loop_util",
    "_synchronize_device_for_metrics": "loop_util",
    "_initialize_data_worker": "loop_util",
}


class ImportParityTests(unittest.TestCase):
    def test_shim_all_matches_owner_map(self) -> None:
        self.assertEqual(set(SHIM_OWNER), set(getattr(trainer, "__all__", SHIM_OWNER)))

    def test_every_symbol_resolves_to_the_same_object(self) -> None:
        for name, owner in SHIM_OWNER.items():
            with self.subTest(name=name, owner=owner):
                owner_module = __import__(
                    f"game_cls.engine.training.{owner}",
                    fromlist=[name],
                )
                self.assertIs(
                    getattr(trainer, name),
                    getattr(owner_module, name),
                    f"{name!r} differs between the trainer shim and "
                    f"engine.training.{owner}",
                )

    def test_run_training_globals_point_at_the_loop_module(self) -> None:
        # The training loop must resolve its helpers from the loop module's
        # own namespace; otherwise call-site patches (e.g. _run_evaluation)
        # silently miss and tests pass for the wrong reason.
        self.assertIs(trainer.run_training.__globals__, loop.__dict__)


if __name__ == "__main__":
    unittest.main()
