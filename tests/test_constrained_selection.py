"""Constrained hierarchical model selection (step4 §三.3).

Unit-tests the selection helpers directly with hand-built metrics dicts:
Layer-1 FPR/recall gates, Layer-2 recall -> worst-game recall -> negative
score p99.9 ranking, and the unchanged metric/composite modes.
"""

from __future__ import annotations

import json
import unittest

from game_cls.engine.trainer import (
    _is_better_model,
    _selection_eligible,
    _selection_rank_key,
)
from game_cls.engine.training.selection import (
    selection_report_fields,
    selection_sort_value,
)


def _constrained_cfg(**overrides) -> dict:
    cfg = {
        "selection_mode": "constrained",
        "max_global_fpr": 0.01,
        "max_worst_game_fpr": 0.02,
        "min_positive_recall": 0.8,
        "minimum_worst_game_f1": None,
    }
    cfg.update(overrides)
    return cfg


def _metrics(
    global_fpr: float = 0.005,
    worst_game_fpr: float = 0.01,
    global_recall: float = 0.9,
    worst_game_recall: float = 0.8,
    negative_p999: float = 0.10,
    worst_game_f1: float = 0.7,
) -> dict:
    return {
        "global_fpr_at_decision_threshold": global_fpr,
        "worst_game_fpr_at_decision_threshold": worst_game_fpr,
        "global_positive_recall_at_decision_threshold": global_recall,
        "worst_game_positive_recall_at_decision_threshold": worst_game_recall,
        "negative_score_p999": negative_p999,
        "worst_game_f1_at_decision_threshold": worst_game_f1,
    }


class ConstrainedEligibilityTests(unittest.TestCase):
    def test_all_gates_pass(self) -> None:
        self.assertTrue(_selection_eligible(_metrics(), _constrained_cfg()))

    def test_global_fpr_gate_rejects_even_with_high_recall(self) -> None:
        cfg = _constrained_cfg()
        self.assertFalse(
            _selection_eligible(_metrics(global_fpr=0.05, global_recall=0.99), cfg)
        )
        self.assertTrue(
            _selection_eligible(_metrics(global_fpr=0.01, global_recall=0.99), cfg)
        )

    def test_worst_game_fpr_gate(self) -> None:
        cfg = _constrained_cfg()
        self.assertFalse(_selection_eligible(_metrics(worst_game_fpr=0.03), cfg))
        self.assertTrue(_selection_eligible(_metrics(worst_game_fpr=0.02), cfg))

    def test_min_positive_recall_gate(self) -> None:
        cfg = _constrained_cfg()
        self.assertFalse(_selection_eligible(_metrics(global_recall=0.79), cfg))
        self.assertTrue(_selection_eligible(_metrics(global_recall=0.80), cfg))

    def test_null_gates_are_skipped(self) -> None:
        cfg = _constrained_cfg(
            max_global_fpr=None,
            max_worst_game_fpr=None,
            min_positive_recall=None,
        )
        self.assertTrue(
            _selection_eligible(
                _metrics(
                    global_fpr=0.5,
                    worst_game_fpr=0.9,
                    global_recall=0.1,
                ),
                cfg,
            )
        )

    def test_missing_metric_fails_the_gate(self) -> None:
        # A candidate missing the required metric cannot be eligible.
        self.assertFalse(_selection_eligible({}, _constrained_cfg()))

    def test_minimum_worst_game_f1_applies_as_extra_gate(self) -> None:
        cfg = _constrained_cfg(minimum_worst_game_f1=0.5)
        self.assertFalse(_selection_eligible(_metrics(worst_game_f1=0.4), cfg))
        self.assertTrue(_selection_eligible(_metrics(worst_game_f1=0.5), cfg))


class ConstrainedRankingTests(unittest.TestCase):
    def test_rank_key_orders_recall_worst_recall_negative_p999(self) -> None:
        cfg = _constrained_cfg()
        # negative_p999 values are probabilities in [0,1]; lower = better.
        high_recall = _metrics(
            global_recall=0.9, worst_game_recall=0.8, negative_p999=0.10
        )
        lower_recall = _metrics(
            global_recall=0.85, worst_game_recall=0.8, negative_p999=0.10
        )
        lower_worst = _metrics(
            global_recall=0.9, worst_game_recall=0.7, negative_p999=0.10
        )
        higher_p999 = _metrics(
            global_recall=0.9, worst_game_recall=0.8, negative_p999=0.27
        )
        self.assertGreater(
            _selection_rank_key(high_recall, cfg),
            _selection_rank_key(lower_recall, cfg),
        )
        self.assertGreater(
            _selection_rank_key(high_recall, cfg),
            _selection_rank_key(lower_worst, cfg),
        )
        self.assertGreater(
            _selection_rank_key(high_recall, cfg),
            _selection_rank_key(higher_p999, cfg),
        )
        self.assertEqual(
            _selection_rank_key(high_recall, cfg),
            _selection_rank_key(dict(high_recall), cfg),
        )

    def test_better_model_prefers_higher_global_recall(self) -> None:
        cfg = _constrained_cfg()
        better = _metrics(global_recall=0.9)
        worse = _metrics(global_recall=0.85)
        self.assertTrue(_is_better_model(better, worse, cfg))
        self.assertFalse(_is_better_model(worse, better, cfg))

    def test_better_model_breaks_recall_tie_on_worst_game_recall(self) -> None:
        cfg = _constrained_cfg()
        better = _metrics(global_recall=0.9, worst_game_recall=0.8)
        worse = _metrics(global_recall=0.9, worst_game_recall=0.7)
        self.assertTrue(_is_better_model(better, worse, cfg))
        self.assertFalse(_is_better_model(worse, better, cfg))

    def test_better_model_breaks_tie_on_negative_p999(self) -> None:
        cfg = _constrained_cfg()
        # Lower probability = fewer hard negatives near the threshold = better.
        better = _metrics(
            global_recall=0.9,
            worst_game_recall=0.8,
            negative_p999=0.05,
        )
        worse = _metrics(
            global_recall=0.9,
            worst_game_recall=0.8,
            negative_p999=0.27,
        )
        self.assertTrue(_is_better_model(better, worse, cfg))
        self.assertFalse(_is_better_model(worse, better, cfg))

    def test_ineligible_candidate_never_beats_anything(self) -> None:
        cfg = _constrained_cfg()
        ineligible = _metrics(global_fpr=0.9)
        eligible = _metrics()
        self.assertFalse(_is_better_model(ineligible, {}, cfg))
        self.assertFalse(_is_better_model(ineligible, eligible, cfg))

    def test_empty_incumbent_is_beaten_by_eligible_candidate(self) -> None:
        cfg = _constrained_cfg()
        self.assertTrue(_is_better_model(_metrics(), {}, cfg))
        self.assertFalse(_is_better_model(_metrics(global_fpr=0.9), {}, cfg))

    def test_ineligible_incumbent_is_replaced(self) -> None:
        cfg = _constrained_cfg()
        eligible = _metrics()
        incumbent = _metrics(global_recall=0.5)  # below min_positive_recall
        self.assertTrue(_is_better_model(eligible, incumbent, cfg))


class SelectionSortValueTests(unittest.TestCase):
    """selection_sort_value is the one ordering every consumer must reuse."""

    def test_constrained_key_is_a_json_round_trippable_list(self) -> None:
        cfg = _constrained_cfg()
        value = selection_sort_value(_metrics(), cfg)
        self.assertIsInstance(value, list)
        # eligibility flag + the three constrained components.
        # negative_score_p999 is now stored as probability [0,1]; the sort
        # component is negated so lower probability ranks higher.
        self.assertEqual(value, [1.0, 0.9, 0.8, -0.1])
        # A tuple would decode back as a list and stop comparing; a list
        # survives the topk registry / run index round-trip unchanged.
        self.assertEqual(json.loads(json.dumps(value)), value)

    def test_bare_key_drops_the_eligibility_prefix(self) -> None:
        cfg = _constrained_cfg()
        self.assertEqual(
            selection_sort_value(_metrics(), cfg, include_eligibility=False),
            [0.9, 0.8, -0.1],
        )

    def test_ineligible_sorts_below_every_eligible_candidate(self) -> None:
        cfg = _constrained_cfg()
        # Best scalar recall in the fixture, but the global FPR gate fails.
        ineligible = selection_sort_value(
            _metrics(global_fpr=0.9, global_recall=0.99), cfg
        )
        eligible = selection_sort_value(_metrics(global_recall=0.81), cfg)
        self.assertLess(ineligible, eligible)
        self.assertEqual(ineligible[0], 0.0)
        self.assertEqual(eligible[0], 1.0)

    def test_order_matches_is_better_model(self) -> None:
        cfg = _constrained_cfg()
        # Equal global recall, so the scalar selection_score cannot separate
        # these two at all; only the worst-game recall / negative-p99.9
        # tie-breakers can.
        scalar_winner = _metrics(
            global_recall=0.9, worst_game_recall=0.60, negative_p999=0.27
        )
        tuple_winner = _metrics(
            global_recall=0.9, worst_game_recall=0.85, negative_p999=0.05
        )
        self.assertTrue(_is_better_model(tuple_winner, scalar_winner, cfg))
        self.assertGreater(
            selection_sort_value(tuple_winner, cfg),
            selection_sort_value(scalar_winner, cfg),
        )
        # The scalar selection_score cannot see the difference at all.
        self.assertEqual(
            scalar_winner["global_positive_recall_at_decision_threshold"],
            tuple_winner["global_positive_recall_at_decision_threshold"],
        )

    def test_metric_mode_key_is_a_single_component(self) -> None:
        cfg = {"selection_metric": "global_f1_at_decision_threshold"}
        candidate = {"global_f1_at_decision_threshold": 0.77}
        self.assertEqual(selection_sort_value(candidate, cfg), [1.0, 0.77])
        self.assertEqual(
            selection_sort_value(candidate, cfg, include_eligibility=False), [0.77]
        )

    def test_report_fields_expose_eligibility_and_tie_breakers(self) -> None:
        cfg = _constrained_cfg()
        fields = selection_report_fields(
            _metrics(global_recall=0.88, worst_game_recall=0.72, negative_p999=0.08),
            cfg,
        )
        self.assertEqual(fields["selection_mode"], "constrained")
        self.assertTrue(fields["selection_eligible"])
        self.assertAlmostEqual(fields["selection_score"], 0.88)
        self.assertAlmostEqual(fields["worst_game_positive_recall"], 0.72)
        self.assertAlmostEqual(fields["negative_score_p999"], 0.08)
        self.assertEqual(fields["selection_sort_value"], [1.0, 0.88, 0.72, -0.08])
        # JSON-serializable so it can live in index.jsonl.
        self.assertEqual(json.loads(json.dumps(fields)), fields)

    def test_report_fields_record_ineligibility(self) -> None:
        cfg = _constrained_cfg()
        fields = selection_report_fields(_metrics(global_fpr=0.5), cfg)
        self.assertFalse(fields["selection_eligible"])
        self.assertAlmostEqual(fields["global_fpr"], 0.5)


class MetricAndCompositeSelectionTests(unittest.TestCase):
    def test_metric_mode_unchanged(self) -> None:
        cfg = {
            "selection_mode": "metric",
            "selection_metric": "global_f1_at_decision_threshold",
            "minimum_worst_game_f1": 0.5,
        }
        candidate = {
            "global_f1_at_decision_threshold": 0.9,
            "worst_game_f1_at_decision_threshold": 0.6,
        }
        self.assertTrue(_selection_eligible(candidate, cfg))
        self.assertFalse(
            _selection_eligible(
                dict(candidate, worst_game_f1_at_decision_threshold=0.4),
                cfg,
            )
        )
        # Scalar rank key, not the constrained tuple.
        self.assertEqual(_selection_rank_key(candidate, cfg), 0.9)
        better = dict(candidate, global_f1_at_decision_threshold=0.95)
        self.assertTrue(_is_better_model(better, candidate, cfg))

    def test_metric_mode_without_explicit_mode_defaults_to_metric(self) -> None:
        cfg = {"selection_metric": "global_f1_tau099"}
        candidate = {"global_f1_tau099": 0.88}
        self.assertTrue(_selection_eligible(candidate, cfg))
        self.assertEqual(_selection_rank_key(candidate, cfg), 0.88)

    def test_composite_mode_unchanged(self) -> None:
        cfg = {
            "selection_mode": "composite",
            "selection_metric": "composite",
            "selection_weights": {
                "global_f1": 0.4,
                "macro_game_f1": 0.4,
                "worst_game_f1": 0.2,
            },
            "minimum_worst_game_f1": 0.5,
        }
        candidate = {
            "global_f1_at_decision_threshold": 0.9,
            "macro_game_f1_at_decision_threshold": 0.8,
            "worst_game_f1_at_decision_threshold": 0.6,
        }
        self.assertTrue(_selection_eligible(candidate, cfg))
        self.assertAlmostEqual(_selection_rank_key(candidate, cfg), 0.8)
        better = dict(
            candidate,
            global_f1_at_decision_threshold=0.95,
        )
        self.assertTrue(_is_better_model(better, candidate, cfg))
        rejected = dict(candidate, worst_game_f1_at_decision_threshold=0.4)
        self.assertFalse(_is_better_model(rejected, {}, cfg))

    def test_composite_implied_by_selection_metric_without_mode(self) -> None:
        cfg = {
            "selection_metric": "composite",
            "selection_weights": {
                "global_f1": 1.0,
                "macro_game_f1": 0.0,
                "worst_game_f1": 0.0,
            },
        }
        candidate = {"global_f1_at_decision_threshold": 0.7}
        self.assertTrue(_selection_eligible(candidate, cfg))
        self.assertAlmostEqual(_selection_rank_key(candidate, cfg), 0.7)


if __name__ == "__main__":
    unittest.main()
