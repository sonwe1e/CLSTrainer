"""Benchmark gate checks and report writing (step5 P3/P6, step6 gate contract).

The gate contract is ``{metric_name: {op, value}}`` with bare scalars kept for
backward compatibility. The direction of a bare scalar comes from an explicit
table, never from substrings of the metric name: the old ``"fpr" in name``
heuristic upper-bounded specificity (higher is better) and lower-bounded ECE
and the negative-score percentiles (lower is better), so those three cases get
fixtures whose value verdict *flips* between ``<=`` and ``>=``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game_cls.reports.benchmark import (
    GATE_OPERATORS,
    check_gates,
    gateable_metric_names,
    validate_gate_metrics,
    write_benchmark_report,
)


class BenchmarkGateTests(unittest.TestCase):
    def test_fpr_gate_uses_upper_bound(self) -> None:
        metrics = {"global_fpr_at_decision_threshold": 0.008}
        checks = check_gates(metrics, {"global_fpr_at_decision_threshold": 0.01})
        self.assertTrue(checks[0][1])
        metrics["global_fpr_at_decision_threshold"] = 0.02
        checks = check_gates(metrics, {"global_fpr_at_decision_threshold": 0.01})
        self.assertFalse(checks[0][1])

    def test_recall_gate_uses_lower_bound(self) -> None:
        metrics = {"global_positive_recall_at_decision_threshold": 0.85}
        checks = check_gates(
            metrics, {"global_positive_recall_at_decision_threshold": 0.8}
        )
        self.assertTrue(checks[0][1])
        metrics["global_positive_recall_at_decision_threshold"] = 0.7
        checks = check_gates(
            metrics, {"global_positive_recall_at_decision_threshold": 0.8}
        )
        self.assertFalse(checks[0][1])

    def test_absent_metric_fails_gate(self) -> None:
        # A producible metric this run did not emit (no subtype grouping).
        checks = check_gates({}, {"worst_subtype_fpr_at_decision_threshold": 0.01})
        self.assertFalse(checks[0][1])
        self.assertIn("metric absent", checks[0][2])

    def test_unknown_metric_name_raises_instead_of_silently_failing(self) -> None:
        # An unpassable-by-construction gate must not read as a normal
        # "metric absent" failure that a human would blame on the model.
        with self.assertRaisesRegex(ValueError, "not a metric the evaluator"):
            check_gates({}, {"missing_metric": 0.01})

    def test_specificity_scalar_is_a_lower_bound(self) -> None:
        """The old substring guess had no 'specificity' rule but its own
        comment claimed one; a higher-is-better metric must never be upper
        bounded. value=0.97 bound=0.95 passes as ">=" and fails as "<=".
        """
        checks = check_gates(
            {"global_specificity_at_decision_threshold": 0.97},
            {"global_specificity_at_decision_threshold": 0.95},
        )
        self.assertTrue(checks[0][1], checks[0][2])
        self.assertIn(">=", checks[0][2])
        # And the direction really bites: below the floor it fails.
        checks = check_gates(
            {"global_specificity_at_decision_threshold": 0.90},
            {"global_specificity_at_decision_threshold": 0.95},
        )
        self.assertFalse(checks[0][1], checks[0][2])

    def test_ece_scalar_is_an_upper_bound(self) -> None:
        """Calibration error is lower-better; the substring guess had no
        'ece' rule and would have read 0.02 >= 0.05 as a failure.
        """
        checks = check_gates({"ece_tail_95_100": 0.02}, {"ece_tail_95_100": 0.05})
        self.assertTrue(checks[0][1], checks[0][2])
        self.assertIn("<=", checks[0][2])
        checks = check_gates({"ece_tail_95_100": 0.11}, {"ece_tail_95_100": 0.05})
        self.assertFalse(checks[0][1], checks[0][2])

    def test_negative_score_p999_scalar_is_an_upper_bound(self) -> None:
        """A hard negative scoring 0.995 at a 0.99 bound must FAIL. The
        substring guess would have read it as value >= bound -> pass.
        """
        checks = check_gates(
            {"negative_score_p999": 0.995}, {"negative_score_p999": 0.99}
        )
        self.assertFalse(checks[0][1], checks[0][2])
        checks = check_gates(
            {"negative_score_p999": 0.80}, {"negative_score_p999": 0.99}
        )
        self.assertTrue(checks[0][1], checks[0][2])

    def test_explicit_op_overrides_the_natural_direction(self) -> None:
        # Same metric, same value, opposite verdicts -- the operator decides.
        metrics = {"global_fpr_at_decision_threshold": 0.02}
        upper = check_gates(
            metrics, {"global_fpr_at_decision_threshold": {"op": "<=", "value": 0.01}}
        )
        lower = check_gates(
            metrics, {"global_fpr_at_decision_threshold": {"op": ">=", "value": 0.01}}
        )
        self.assertFalse(upper[0][1], upper[0][2])
        self.assertTrue(lower[0][1], lower[0][2])

    def test_strict_operators_are_distinct_from_inclusive_ones(self) -> None:
        metrics = {"global_fpr_at_decision_threshold": 0.01}
        inclusive = check_gates(
            metrics, {"global_fpr_at_decision_threshold": {"op": "<=", "value": 0.01}}
        )
        strict = check_gates(
            metrics, {"global_fpr_at_decision_threshold": {"op": "<", "value": 0.01}}
        )
        self.assertTrue(inclusive[0][1])
        self.assertFalse(strict[0][1])

    def test_unknown_operator_is_an_error(self) -> None:
        for bad_op in ("=<", "==", "lt", "≤", ""):
            with (
                self.subTest(op=bad_op),
                self.assertRaisesRegex(ValueError, "not a comparison operator"),
            ):
                check_gates(
                    {"global_fpr_at_decision_threshold": 0.5},
                    {
                        "global_fpr_at_decision_threshold": {
                            "op": bad_op,
                            "value": 0.01,
                        }
                    },
                )

    def test_malformed_explicit_gate_is_an_error(self) -> None:
        cases = (
            ({"value": 0.01}, "must set both"),
            ({"op": "<="}, "must set both"),
            ({"op": "<=", "value": 0.01, "unit": "%"}, "unknown field"),
            ({"op": "<=", "value": "0.01"}, "must be a number"),
            ({"op": "<=", "value": True}, "must be a number"),
        )
        for spec, expected in cases:
            with (
                self.subTest(spec=spec),
                self.assertRaisesRegex(ValueError, expected),
            ):
                check_gates(
                    {"global_fpr_at_decision_threshold": 0.5},
                    {"global_fpr_at_decision_threshold": spec},
                )

    def test_non_directional_metric_requires_the_explicit_form(self) -> None:
        # sample_count has no better-direction: 2000 could mean "at least" or
        # "at most". Guessing either way would be a silent wrong answer.
        with self.assertRaisesRegex(ValueError, "no documented better-direction"):
            check_gates({"sample_count": 5000}, {"sample_count": 2000})
        checks = check_gates(
            {"sample_count": 5000}, {"sample_count": {"op": ">=", "value": 2000}}
        )
        self.assertTrue(checks[0][1])

    def test_legacy_tau099_alias_satisfies_a_neutral_gate(self) -> None:
        # An older run's metrics dict only has the _tau099 name.
        checks = check_gates(
            {"worst_game_f1_tau099": 0.82},
            {"worst_game_f1_at_decision_threshold": 0.80},
        )
        self.assertTrue(checks[0][1], checks[0][2])
        # ...and the reverse direction of the alias map works too.
        checks = check_gates(
            {"global_f1_at_decision_threshold": 0.60},
            {"global_f1_tau099": 0.80},
        )
        self.assertFalse(checks[0][1], checks[0][2])


class GateMetricValidationTests(unittest.TestCase):
    def test_empty_gate_dict_has_no_problems(self) -> None:
        self.assertEqual(validate_gate_metrics(None), [])
        self.assertEqual(validate_gate_metrics({}), [])

    def test_typo_is_reported_with_a_suggestion(self) -> None:
        problems = validate_gate_metrics({"global_fpr_at_decision_threshhold": 0.01})
        self.assertEqual(len(problems), 1)
        self.assertIn("global_fpr_at_decision_threshhold", problems[0])
        self.assertIn("Did you mean 'global_fpr_at_decision_threshold'", problems[0])

    def test_documented_schema_example_keys_are_rejected(self) -> None:
        """The keys the schema example used to document (max_global_fpr /
        min_positive_recall) are selection knobs, not evaluator metrics; a
        gate naming them could never pass and must fail at config time.
        """
        problems = validate_gate_metrics(
            {"max_global_fpr": 0.01, "min_positive_recall": 0.8}
        )
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("could never pass" in problem for problem in problems))

    def test_bad_operator_and_bad_scalar_are_reported(self) -> None:
        problems = validate_gate_metrics(
            {
                "global_fpr_at_decision_threshold": {"op": "=<", "value": 0.01},
                "sample_count": 2000,
            }
        )
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("not a comparison operator" in p for p in problems))
        self.assertTrue(any("no documented better-direction" in p for p in problems))

    def test_non_mapping_gate_block_is_reported(self) -> None:
        self.assertEqual(len(validate_gate_metrics([1, 2])), 1)

    def test_every_operator_in_the_contract_validates(self) -> None:
        for op in GATE_OPERATORS:
            with self.subTest(op=op):
                self.assertEqual(
                    validate_gate_metrics(
                        {"global_fpr_at_decision_threshold": {"op": op, "value": 0.5}}
                    ),
                    [],
                )

    def test_release_recipe_gates_validate(self) -> None:
        from game_cls.config import load_config

        config = load_config("configs/recipes/game_cls_release.yaml")
        gate_metrics = config["benchmark"]["gate_metrics"]
        self.assertTrue(gate_metrics)
        self.assertEqual(validate_gate_metrics(gate_metrics), [])
        # The shipped example uses the explicit form throughout.
        for name, spec in gate_metrics.items():
            with self.subTest(metric=name):
                self.assertIsInstance(spec, dict, name)
                self.assertIn(spec["op"], GATE_OPERATORS)


class GateMetricRegistryDriftTests(unittest.TestCase):
    """A gate name must be producible, so the registry cannot drift away
    from what ``engine.evaluator.evaluate`` really emits."""

    def _evaluated_metric_names(self) -> set[str]:
        import tests.test_evaluator_subtype_end_to_end as e2e

        torch = e2e.torch
        if torch is None:
            self.skipTest("torch is not installed")
        from game_cls.engine.evaluator import evaluate

        batch = e2e._batch(
            subtype_ids=[0, 0, 1, 1, 2, 2],
            labels=[0, 0, 0, 0, 1, 1],
            video_ids=[0, 0, 1, 1, 2, 2],
        )
        model = e2e._ConstantMargin()
        model._margins = torch.tensor([-4.0, -4.0, 4.0, 4.0, 4.0, 4.0])
        result = evaluate(
            model,
            e2e._Loader([batch], e2e.CATALOGS),
            torch.device("cpu"),
            0.5,
            group_catalogs=e2e.CATALOGS,
            evaluation_kind="full",
            full_auc_mode="histogram",
        )
        return {
            name
            for name, value in (result.metrics or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def test_registry_matches_a_full_evaluation_exactly(self) -> None:
        """A full evaluation with group catalogs is exactly the shape the
        benchmark path runs, so its numeric metrics and the gateable registry
        must be the same set -- no unproducible gate names, and no new
        evaluator metric that silently cannot be gated.
        """
        produced = self._evaluated_metric_names()
        registry = set(gateable_metric_names())
        self.assertEqual(
            produced,
            registry,
            "gate registry drifted from evaluator metrics; "
            f"unproducible gate names: {sorted(registry - produced)}; "
            "evaluator metrics missing a direction (add to _LOWER_BETTER / "
            f"_HIGHER_BETTER / _NON_DIRECTIONAL): {sorted(produced - registry)}",
        )

    def test_every_produced_metric_has_exactly_one_direction(self) -> None:
        from game_cls.reports.benchmark import (
            _HIGHER_BETTER,
            _LOWER_BETTER,
            _NON_DIRECTIONAL,
        )

        # Overlapping tables would make a bare scalar's direction depend on
        # the order the tables are consulted.
        self.assertEqual(_LOWER_BETTER & _HIGHER_BETTER, frozenset())
        self.assertEqual(_LOWER_BETTER & _NON_DIRECTIONAL, frozenset())
        self.assertEqual(_HIGHER_BETTER & _NON_DIRECTIONAL, frozenset())


class BenchmarkReportTests(unittest.TestCase):
    def test_report_written_with_scores_and_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = write_benchmark_report(
                Path(directory),
                run_id="run-1",
                checkpoint_alias="best_selection",
                metrics={
                    "sample_count": 100,
                    "global_fpr_at_decision_threshold": 0.005,
                },
                gates=[
                    ("global_fpr_at_decision_threshold", True, "value=0.005 bound=0.01")
                ],
                grouped_metrics=None,
            )
            self.assertTrue(report.is_file())
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run-1")
            self.assertTrue(payload["gates"][0]["passed"])

    def test_report_records_the_gate_spec_and_every_gated_value(self) -> None:
        """``config validate --release`` re-judges a report, so the report has
        to carry the raw gate spec and a value for each gated metric even when
        the metric is outside the fixed summary."""
        gate_metrics = {
            "ece_tail_95_100": {"op": "<=", "value": 0.05},
            "specificity": 0.95,
        }
        metrics = {
            "sample_count": 100,
            "global_fpr_at_decision_threshold": 0.005,
            "ece_tail_95_100": 0.02,
            "specificity": 0.97,
        }
        with tempfile.TemporaryDirectory() as directory:
            report = write_benchmark_report(
                Path(directory),
                run_id="run-2",
                checkpoint_alias="best_selection",
                metrics=metrics,
                gates=check_gates(metrics, gate_metrics),
                grouped_metrics=None,
                gate_metrics=gate_metrics,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["gate_metrics"], gate_metrics)
            self.assertAlmostEqual(payload["scores"]["ece_tail_95_100"], 0.02)
            self.assertAlmostEqual(payload["scores"]["specificity"], 0.97)
            # The recorded scores alone must be enough to re-judge the gates.
            rejudged = check_gates(payload["scores"], payload["gate_metrics"])
            self.assertTrue(all(passed for _, passed, _ in rejudged), rejudged)


class ReleaseGateVerificationTests(unittest.TestCase):
    """``config validate --release`` must judge a real report, not just the
    presence of a non-empty gate dict."""

    def _validate(self, base: Path, *, gate_metrics: dict) -> tuple[int, str, str]:
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        import yaml

        from game_cls.cli.config_tools import cmd_config_validate

        config = yaml.safe_load(
            Path("configs/recipes/game_cls_production.yaml").read_text(encoding="utf-8")
        )
        config.setdefault("evaluation", {})["minimum_worst_game_f1"] = 0.75
        config.setdefault("benchmark", {})["gate_metrics"] = gate_metrics
        config["benchmark"]["output_dir"] = str(base / "benchmarks")
        cfg_path = base / "release.yaml"
        cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), "utf-8")
        args = type(
            "Args", (), {"config": str(cfg_path), "overrides": [], "release": True}
        )()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), patch("sys.stderr", err):
            code = cmd_config_validate(args)
        return code, out.getvalue(), err.getvalue()

    def _report(self, base: Path, *, metrics: dict, gate_metrics: dict) -> Path:
        return write_benchmark_report(
            base / "benchmarks",
            run_id="run-1",
            checkpoint_alias="best_selection",
            metrics=metrics,
            gates=check_gates(metrics, gate_metrics),
            grouped_metrics=None,
            gate_metrics=gate_metrics,
        )

    def test_missing_report_is_a_note_not_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, out, _ = self._validate(
                Path(directory),
                gate_metrics={"global_fpr_at_decision_threshold": 0.01},
            )
            self.assertEqual(code, 0)
            self.assertIn("UNVERIFIED", out)

    def test_unproducible_gate_name_fails_validation(self) -> None:
        # min_positive_recall is a selection knob, not an evaluator metric.
        # semantic_validate rejects it, so the command never reaches the
        # report lookup -- the point is that it fails with a real reason.
        with tempfile.TemporaryDirectory() as directory:
            code, _, err = self._validate(
                Path(directory),
                gate_metrics={"min_positive_recall": 0.8},
            )
            self.assertEqual(code, 2)
            self.assertIn("could never pass", err)

    def test_unknown_operator_fails_the_release_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, _, err = self._validate(
                Path(directory),
                gate_metrics={
                    "global_fpr_at_decision_threshold": {"op": "=<", "value": 0.01}
                },
            )
            self.assertEqual(code, 2)
            self.assertIn("not a comparison operator", err)

    def test_passing_report_verifies_the_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            gate_metrics = {"global_fpr_at_decision_threshold": 0.01}
            self._report(
                base,
                metrics={
                    "sample_count": 100,
                    "global_fpr_at_decision_threshold": 0.004,
                },
                gate_metrics=gate_metrics,
            )
            code, out, err = self._validate(base, gate_metrics=gate_metrics)
            self.assertEqual(code, 0, err)
            self.assertIn("report.json", out)
            self.assertNotIn("UNVERIFIED", out)

    def test_failing_report_fails_the_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            gate_metrics = {"global_fpr_at_decision_threshold": 0.01}
            self._report(
                base,
                metrics={"sample_count": 100, "global_fpr_at_decision_threshold": 0.09},
                gate_metrics=gate_metrics,
            )
            code, _, err = self._validate(base, gate_metrics=gate_metrics)
            self.assertEqual(code, 2)
            self.assertIn("fails gate global_fpr_at_decision_threshold", err)

    def test_stale_report_cannot_verify_a_new_gate(self) -> None:
        """A report measured under an older gate set never saw the new metric,
        so calling the release verified would be a lie."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self._report(
                base,
                metrics={
                    "sample_count": 100,
                    "global_fpr_at_decision_threshold": 0.004,
                },
                gate_metrics={"global_fpr_at_decision_threshold": 0.01},
            )
            code, _, err = self._validate(
                base,
                gate_metrics={
                    "global_fpr_at_decision_threshold": 0.01,
                    "ece_tail_95_100": {"op": "<=", "value": 0.05},
                },
            )
            self.assertEqual(code, 2)
            self.assertIn("predates the current gates", err)
            self.assertIn("ece_tail_95_100", err)

    def test_report_recorded_under_the_old_direction_is_re_judged(self) -> None:
        """The report's own ``passed`` flags are not trusted: a specificity
        gate that the old substring guess recorded as passing (0.90 <= 0.95)
        must be re-checked as ">=" and fail."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            report_dir = base / "benchmarks" / "run-1_best_selection"
            report_dir.mkdir(parents=True)
            (report_dir / "report.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "checkpoint": "best_selection",
                        "scores": {"global_specificity_at_decision_threshold": 0.90},
                        "gate_metrics": {
                            "global_specificity_at_decision_threshold": 0.95
                        },
                        "gates": [
                            {
                                "name": "global_specificity_at_decision_threshold",
                                "passed": True,
                                "detail": "value=0.9 bound=0.95",
                            }
                        ],
                        "grouped": {},
                    }
                ),
                encoding="utf-8",
            )
            code, _, err = self._validate(
                base,
                gate_metrics={"global_specificity_at_decision_threshold": 0.95},
            )
            self.assertEqual(code, 2)
            self.assertIn("fails gate global_specificity_at_decision_threshold", err)


if __name__ == "__main__":
    unittest.main()
