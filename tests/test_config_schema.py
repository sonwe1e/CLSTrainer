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
        config = load_config("configs/recipes/example_debug.yaml")
        config["optimizer"]["lerning_rate"] = 0.1
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(config)
        message = str(ctx.exception)
        self.assertIn("optimizer.lerning_rate", message)
        self.assertIn("Did you mean 'learning_rate'", message)

    def test_unsupported_key_is_unknown(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["optimizer"]["name"] = "AdamW"
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(config)
        message = str(ctx.exception)
        self.assertIn("Unknown config key: optimizer.name", message)

    def test_save_all_errors_is_rejected(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["evaluation"]["save_all_errors"] = True
        with self.assertRaisesRegex(ConfigSchemaError, "save_all_errors"):
            finalize_config(config)

    def test_type_mismatch_is_rejected(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["train"]["local_batch_size"] = "sixty-four"
        with self.assertRaisesRegex(ConfigSchemaError, "local_batch_size"):
            finalize_config(config)

    def test_enum_violation_is_rejected(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["checkpoint"]["periodic_state_mode"] = "sometimes"
        with self.assertRaisesRegex(ConfigSchemaError, "periodic_state_mode"):
            finalize_config(config)

    def test_all_shipped_configs_validate(self) -> None:
        for path in (
            "configs/recipes/example_debug.yaml",
            "configs/recipes/game_cls_production.yaml",
            "configs/recipes/game_cls_release.yaml",
            "configs/recipes/npu_synthetic_smoke.yaml",
        ):
            config = load_config(path)
            self.assertIn("decision", config, path)


class OverrideStrictnessTests(unittest.TestCase):
    def test_typo_override_is_rejected_with_suggestion(self) -> None:
        with self.assertRaises(ConfigSchemaError) as ctx:
            load_config(
                "configs/recipes/example_debug.yaml", ["optimzier.learning_rate=0.0001"]
            )
        message = str(ctx.exception)
        self.assertIn("optimzier.learning_rate", message)
        self.assertIn("optimizer.learning_rate", message)

    def test_unsupported_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "Unknown config override"):
            load_config("configs/recipes/example_debug.yaml", ["optimizer.name=SGD"])

    def test_valid_override_applies(self) -> None:
        config = load_config(
            "configs/recipes/example_debug.yaml", ["train.max_steps=2"]
        )
        self.assertEqual(config["train"]["max_steps"], 2)

    def test_schema_known_path_missing_from_file_is_created(self) -> None:
        # cuda_debug.yaml has no dataloader.train section, but the schema
        # knows it, so a CLI override may legitimately create it.
        config = load_config(
            "configs/recipes/example_debug.yaml", ["dataloader.train.num_workers=0"]
        )
        self.assertEqual(config["dataloader"]["train"]["num_workers"], 0)


class SemanticValidationTests(unittest.TestCase):
    """Third layer: ranges, probabilities and cross-field checks (step3 高优-1)."""

    def _load(self, **overrides: str) -> dict:
        items = [f"{key}={value}" for key, value in overrides.items()]
        return load_config("configs/recipes/example_debug.yaml", items)

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

    def test_warmup_exceeding_budget_emits_warning_not_error(self) -> None:
        # P0-1 fix: warmup>budget is numerically safe (scheduler stays in ramp);
        # it is now a warning via schedule_budget_warnings(), not a hard error.
        from game_cls.config_schema import schedule_budget_warnings

        config = self._load(
            **{"scheduler.warmup_steps": "200", "train.max_steps": "100"}
        )
        warnings = schedule_budget_warnings(config)
        self.assertEqual(len(warnings), 1)
        self.assertIn("200", warnings[0])
        self.assertIn("100", warnings[0])

    def test_constrained_mode_rejects_dead_composite_keys(self) -> None:
        # Audit P1-4: under selection_mode=constrained the rank key is
        # (global recall, worst-game recall, -negative p99.9); a composite
        # selection_metric/selection_weights can never influence the best
        # checkpoint, so they must be a config error instead of silent noise.
        with self.assertRaisesRegex(ConfigSchemaError, "constrained"):
            self._load(
                **{
                    "evaluation.selection_mode": "constrained",
                    "evaluation.selection_metric": "composite",
                    "evaluation.selection_weights": (
                        "{global_f1: 0.5, macro_game_f1: 0.5}"
                    ),
                }
            )

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

    # --- Defect 3: mode: min with a selection monitor (step6 remediation) ---

    def test_selection_monitor_with_mode_min_is_rejected(self) -> None:
        """The bad combination must produce a loud error naming both keys."""
        with self.assertRaises(ConfigSchemaError) as ctx:
            self._load(
                **{
                    "early_stopping.monitor": "selection_score",
                    "early_stopping.mode": "min",
                }
            )
        message = str(ctx.exception)
        self.assertIn("early_stopping.monitor", message)
        self.assertIn("early_stopping.mode", message)

    def test_removed_selection_alias_is_rejected(self) -> None:
        """Only the canonical selection_score monitor is accepted."""
        with self.assertRaises(ConfigSchemaError) as ctx:
            self._load(
                **{
                    "early_stopping.monitor": "selection",
                    "early_stopping.mode": "max",
                }
            )
        message = str(ctx.exception)
        self.assertIn("early_stopping.monitor", message)
        self.assertIn("selection_score", message)

    def test_selection_monitor_with_mode_max_is_accepted(self) -> None:
        """monitor: selection_score with mode: max is the legitimate default."""
        # Must not raise; the combination is correct.
        self._load(
            **{
                "early_stopping.monitor": "selection_score",
                "early_stopping.mode": "max",
            }
        )

    def test_non_selection_monitor_with_mode_min_is_accepted(self) -> None:
        """monitor: cross_entropy with mode: min is a valid numeric config."""
        self._load(
            **{
                "early_stopping.monitor": "cross_entropy",
                "early_stopping.mode": "min",
            }
        )


class BenchmarkGateSchemaTests(unittest.TestCase):
    """A gate whose metric name or operator is wrong must fail at config time,
    not read 'metric absent' after a full benchmark run."""

    def _with_gates(self, gate_metrics: dict) -> dict:
        config = load_config("configs/recipes/example_debug.yaml")
        config.setdefault("benchmark", {})["gate_metrics"] = gate_metrics
        return config

    def test_unproducible_metric_name_is_rejected(self) -> None:
        # These are exactly the keys the schema example used to document.
        config = self._with_gates(
            {
                "max_global_fpr": {"op": "<=", "value": 0.01},
                "min_positive_recall": {"op": ">=", "value": 0.8},
            }
        )
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(config)
        message = str(ctx.exception)
        self.assertIn("max_global_fpr", message)
        self.assertIn("could never pass", message)

    def test_unknown_operator_is_rejected(self) -> None:
        config = self._with_gates(
            {"global_fpr_at_decision_threshold": {"op": "=<", "value": 0.01}}
        )
        with self.assertRaisesRegex(ConfigSchemaError, "not a comparison operator"):
            finalize_config(config)

    def test_scalar_gate_is_rejected(self) -> None:
        config = self._with_gates({"sample_count": 2000})
        with self.assertRaisesRegex(ConfigSchemaError, "must use"):
            finalize_config(config)

    def test_explicit_gates_are_accepted(self) -> None:
        config = self._with_gates(
            {
                "global_fpr_at_decision_threshold": {
                    "op": "<=",
                    "value": 0.01,
                },
                "global_positive_recall_at_decision_threshold": {
                    "op": ">=",
                    "value": 0.8,
                },
                "sample_count": {"op": ">=", "value": 2000},
            }
        )
        finalize_config(config)

    def test_release_recipe_validates(self) -> None:
        config = load_config("configs/recipes/game_cls_release.yaml")
        self.assertTrue(config["data"]["hard_negative"]["enabled"])
        self.assertTrue(config["model"]["trainable_rules"])
        self.assertTrue(config["benchmark"]["gate_metrics"])


class HardNegativeSchemaTests(unittest.TestCase):
    """Structural half of the hard-negative contract. This layer is
    filesystem-free by design, so it can only check the config itself; the
    sidecar's existence and content are checked by
    ``data.sidecar.check_hard_negative_readiness``."""

    def _enabled(self, **overrides) -> dict:
        config = load_config("configs/recipes/example_debug.yaml")
        config["data"]["metadata_sidecar"] = "indexes/video_metadata.parquet"
        hard_negative = config["data"].setdefault("hard_negative", {})
        hard_negative.update(
            {
                "enabled": True,
                "subtype_field": "negative_subtype",
                "hard_subtypes": ["wooden_bridge"],
                "ordinary_subtypes": [],
                "negative_mix": {"ordinary": 0.5, "hard": 0.5},
                "min_videos_per_subtype_bucket": 1,
            }
        )
        hard_negative.update(overrides)
        return config

    def test_enabled_without_sidecar_is_rejected(self) -> None:
        config = self._enabled()
        config["data"]["metadata_sidecar"] = None
        with self.assertRaisesRegex(ConfigSchemaError, "metadata_sidecar"):
            finalize_config(config)

    def test_enabled_without_hard_subtypes_is_rejected(self) -> None:
        # Without hard subtypes every negative lands in the ordinary bucket:
        # the config asks for hard-negative mixing and gets plain sampling.
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(self._enabled(hard_subtypes=[]))
        message = str(ctx.exception)
        self.assertIn("hard_subtypes", message)
        self.assertIn("degrades to ordinary negatives", message)

    def test_disabled_without_hard_subtypes_is_fine(self) -> None:
        config = self._enabled(enabled=False, hard_subtypes=[])
        config["data"]["metadata_sidecar"] = None
        finalize_config(config)

    def test_enabled_with_hard_subtypes_is_accepted(self) -> None:
        finalize_config(self._enabled())

    def test_schema_layer_does_not_touch_the_filesystem(self) -> None:
        # A non-existent sidecar path must still pass this layer, otherwise
        # `config show` and every shipped-config test would need real parquet
        # files on disk. The filesystem check lives elsewhere on purpose.
        config = self._enabled()
        config["data"]["metadata_sidecar"] = "indexes/definitely_not_here.parquet"
        finalize_config(config)
        from game_cls.data.sidecar import check_hard_negative_readiness

        self.assertTrue(check_hard_negative_readiness(config))


class DecisionThresholdTests(unittest.TestCase):
    def test_decision_threshold_is_the_single_source(self) -> None:
        config = {
            "decision": {"threshold": 0.9},
            "loss": {},
            "evaluation": {},
        }
        resolve_decision_threshold(config)
        self.assertNotIn("threshold", config["loss"])
        self.assertNotIn("threshold", config["evaluation"])

    def test_removed_threshold_keys_are_rejected(self) -> None:
        config = {
            "decision": {"threshold": 0.99},
            "loss": {"threshold": 0.99},
            "evaluation": {"threshold": 0.99},
        }
        with self.assertRaises(ConfigSchemaError):
            finalize_config(config)

    def test_default_threshold_is_the_business_contract(self) -> None:
        config: dict = {}
        self.assertEqual(resolve_decision_threshold(config), 0.99)
        self.assertEqual(config["decision"]["threshold"], 0.99)

    def test_shipped_configs_expose_decision_threshold(self) -> None:
        config = load_config("configs/recipes/game_cls_production.yaml")
        self.assertEqual(config["decision"]["threshold"], 0.99)
        self.assertNotIn("threshold", config["loss"])
        self.assertNotIn("threshold", config["evaluation"])


class SourceTrackingTests(unittest.TestCase):
    def test_sources_track_recipe_layers(self) -> None:
        _, sources = load_config_with_sources(
            "configs/recipes/game_cls_production.yaml"
        )
        # Machine behavior comes from the profile layer, business facts from
        # the task profile, selection policy from the production preset.
        self.assertTrue(sources["device.accelerator"].endswith("npu_8p.yaml"))
        self.assertTrue(
            sources["decision.threshold"].endswith("dual_frame_binary.yaml")
        )
        self.assertTrue(
            sources["evaluation.selection_mode"].endswith("production.yaml")
        )
        self.assertIn("experiment.seed", sources)

    def test_overrides_are_recorded_as_sources(self) -> None:
        _, sources = load_config_with_sources(
            "configs/recipes/example_debug.yaml", ["train.max_steps=2"]
        )
        self.assertEqual(sources["train.max_steps"], "override:train.max_steps=2")

    def test_known_paths_cover_consumed_keys(self) -> None:
        paths = set(known_dotted_paths())
        for required in (
            "experiment.output_dir",
            "decision.threshold",
            "dataloader.train.num_workers",
            "evaluation.selection_metric",
            "checkpoint.periodic_state_mode",
            "distributed.backend",
            "data.source_video_identity.namespaces.test",
            "data.source_video_identity.namespaces.source",
        ):
            self.assertIn(required, paths)


class SourceNamespaceSchemaTests(unittest.TestCase):
    """step8 source provenance namespaces are schema-known and validated."""

    def _config(self, namespaces: dict) -> dict:
        config = load_config("configs/recipes/example_debug.yaml")
        config["data"]["source_video_identity"]["namespaces"] = namespaces
        return config

    def test_namespaces_block_is_schema_known(self) -> None:
        paths = set(known_dotted_paths())
        for leaf in ("train", "val", "test", "source"):
            self.assertIn(f"data.source_video_identity.namespaces.{leaf}", paths)

    def test_unknown_namespace_key_is_rejected(self) -> None:
        config = self._config({"train": "a", "val": "a", "test": "b", "bogus": "c"})
        with self.assertRaises(ConfigSchemaError) as ctx:
            finalize_config(config)
        self.assertIn("data.source_video_identity.namespaces.bogus", str(ctx.exception))

    def test_valid_explicit_form_finalizes(self) -> None:
        config = self._config(
            {"train": "train_pool", "val": "train_pool", "test": "heldout_pool"}
        )
        finalized = finalize_config(config)
        self.assertEqual(
            finalized["data"]["source_video_identity"]["namespaces"],
            {"train": "train_pool", "val": "train_pool", "test": "heldout_pool"},
        )

    def test_valid_shorthand_form_finalizes(self) -> None:
        config = self._config({"source": "train_pool", "test": "heldout_pool"})
        finalized = finalize_config(config)
        self.assertEqual(
            finalized["data"]["source_video_identity"]["namespaces"]["source"],
            "train_pool",
        )


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
