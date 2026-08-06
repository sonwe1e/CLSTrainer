from __future__ import annotations

import unittest

from game_cls.config import load_config, load_config_with_sources
from game_cls.config_schema import (
    ConfigSchemaError,
    finalize_config,
    known_dotted_paths,
    resolve_decision_threshold,
)


class SchemaStrictnessTests(unittest.TestCase):
    def test_unknown_top_level_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config({"experimnt": {"seed": 1}})
        self.assertIn("Unknown config key: experimnt", str(ctx.exception))
        self.assertIn("experiment", str(ctx.exception))

    def test_unknown_nested_key_is_rejected_with_suggestion(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["optimizer"]["lerning_rate"] = 0.1
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(config)
        message = str(ctx.exception)
        self.assertIn("optimizer.lerning_rate", message)
        self.assertIn("Did you mean 'learning_rate'", message)

    def test_removed_keys_raise_migration_hint(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["optimizer"]["name"] = "AdamW"
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(config)
        message = str(ctx.exception)
        self.assertIn("Removed config key: optimizer.name", message)
        self.assertIn("fixed to AdamW", message)

    def test_save_all_errors_is_rejected(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["evaluation"]["save_all_errors"] = True
        with self.assertRaisesRegex(ConfigSchemaError, "save_all_errors"):
            finalize_config(config)

    def test_type_mismatch_is_rejected(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["train"]["local_batch_size"] = "sixty-four"
        with self.assertRaisesRegex(ConfigSchemaError, "local_batch_size"):
            finalize_config(config)

    def test_enum_violation_is_rejected(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["checkpoint"]["periodic_state_mode"] = "sometimes"
        with self.assertRaisesRegex(ConfigSchemaError, "periodic_state_mode"):
            finalize_config(config)

    def test_all_shipped_configs_validate(self) -> None:
        for path in (
            "configs/cuda_debug.yaml",
            "configs/npu_1p.yaml",
            "configs/npu_8p.yaml",
            "configs/npu_production.yaml",
            "configs/npu_production_packed.yaml",
        ):
            config = load_config(path)
            self.assertIn("decision", config, path)


class OverrideStrictnessTests(unittest.TestCase):
    def test_typo_override_is_rejected_with_suggestion(self) -> None:
        with self.assertRaises(ConfigSchemaError) as ctx:
            load_config("configs/cuda_debug.yaml", ["optimzier.learning_rate=0.0001"])
        message = str(ctx.exception)
        self.assertIn("optimzier.learning_rate", message)
        self.assertIn("optimizer.learning_rate", message)

    def test_removed_key_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ConfigSchemaError, "Removed config key: optimizer.name"
        ):
            load_config("configs/cuda_debug.yaml", ["optimizer.name=SGD"])

    def test_valid_override_applies(self) -> None:
        config = load_config("configs/cuda_debug.yaml", ["train.max_steps=7"])
        self.assertEqual(config["train"]["max_steps"], 7)

    def test_schema_known_path_missing_from_file_is_created(self) -> None:
        # cuda_debug.yaml has no dataloader.train section, but the schema
        # knows it, so a CLI override may legitimately create it.
        config = load_config(
            "configs/cuda_debug.yaml", ["dataloader.train.num_workers=0"]
        )
        self.assertEqual(config["dataloader"]["train"]["num_workers"], 0)


class SemanticValidationTests(unittest.TestCase):
    """Third layer: ranges, probabilities and cross-field checks (step3 高优-1)."""

    def _load(self, **overrides: str) -> dict:
        items = [f"{key}={value}" for key, value in overrides.items()]
        return load_config("configs/cuda_debug.yaml", items)

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "local_batch_size"):
            self._load(**{"train.local_batch_size": "0"})

    def test_negative_learning_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "learning_rate"):
            self._load(**{"optimizer.learning_rate": "-0.001"})

    def test_threshold_outside_unit_interval_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "threshold"):
            self._load(**{"decision.threshold": "1.5"})
        with self.assertRaisesRegex(ConfigSchemaError, "threshold"):
            self._load(**{"decision.threshold": "0"})

    def test_zero_log_every_steps_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "log_every_steps"):
            self._load(**{"train.log_every_steps": "0"})

    def test_probabilities_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "class_probability"):
            self._load(**{"sampler.class_probability": "{0: 1.0, 1: 1.0}"})

    def test_all_zero_probabilities_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "all zero"):
            self._load(**{"sampler.class_probability": "{0: 0.0, 1: 0.0}"})

    def test_warmup_exceeding_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "warmup_steps"):
            self._load(**{"train.warmup_steps": "200", "train.max_steps": "100"})

    def test_negative_selection_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "selection_weights"):
            self._load(
                **{
                    "evaluation.selection_metric": "composite",
                    "evaluation.selection_weights": "{global_f1: -0.5, macro_game_f1: 1.0}",
                }
            )

    def test_valid_config_still_finalizes(self) -> None:
        config = self._load()
        self.assertAlmostEqual(config["decision"]["threshold"], 0.99)


class DecisionThresholdTests(unittest.TestCase):
    def test_decision_threshold_is_the_single_source(self) -> None:
        config = {
            "decision": {"threshold": 0.9},
            "loss": {},
            "evaluation": {},
        }
        resolve_decision_threshold(config)
        self.assertEqual(config["loss"]["threshold"], 0.9)
        self.assertEqual(config["evaluation"]["threshold"], 0.9)

    def test_legacy_keys_agreeing_with_decision_are_tolerated(self) -> None:
        config = {
            "decision": {"threshold": 0.99},
            "loss": {"threshold": 0.99},
            "evaluation": {"threshold": 0.99},
        }
        self.assertEqual(resolve_decision_threshold(config), 0.99)

    def test_conflicting_thresholds_are_rejected(self) -> None:
        config = {
            "decision": {"threshold": 0.99},
            "loss": {"threshold": 0.95},
            "evaluation": {},
        }
        with self.assertRaisesRegex(
            ConfigSchemaError, "Conflicting decision thresholds"
        ):
            resolve_decision_threshold(config)

    def test_default_threshold_is_the_business_contract(self) -> None:
        config: dict = {}
        self.assertEqual(resolve_decision_threshold(config), 0.99)
        self.assertEqual(config["decision"]["threshold"], 0.99)

    def test_shipped_configs_expose_decision_threshold(self) -> None:
        config = load_config("configs/npu_1p.yaml")
        self.assertEqual(config["decision"]["threshold"], 0.99)
        self.assertEqual(config["loss"]["threshold"], 0.99)
        self.assertEqual(config["evaluation"]["threshold"], 0.99)


class SourceTrackingTests(unittest.TestCase):
    def test_sources_track_base_and_child_files(self) -> None:
        _, sources = load_config_with_sources("configs/npu_1p.yaml")
        self.assertTrue(sources["device.accelerator"].endswith("npu_production.yaml"))
        # npu_1p.yaml does not override device.accelerator, so the origin
        # must remain the base file; experiment.output_dir is defined there
        # as well.
        self.assertIn("experiment.seed", sources)

    def test_overrides_are_recorded_as_sources(self) -> None:
        _, sources = load_config_with_sources(
            "configs/cuda_debug.yaml", ["train.max_steps=3"]
        )
        self.assertEqual(sources["train.max_steps"], "override:train.max_steps=3")

    def test_known_paths_cover_consumed_keys(self) -> None:
        paths = set(known_dotted_paths())
        for required in (
            "experiment.output_dir",
            "experiment.run_mode",
            "decision.threshold",
            "dataloader.train.num_workers",
            "evaluation.selection_metric",
            "checkpoint.periodic_state_mode",
            "distributed.backend",
        ):
            self.assertIn(required, paths)


if __name__ == "__main__":
    unittest.main()
