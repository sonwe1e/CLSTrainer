from __future__ import annotations

from typing import Any

# Legacy metric names that predate the neutral "at_decision_threshold"
# naming; used as fallbacks so old configs keep working.
_LEGACY_METRIC_ALIASES = {
    "global_f1_at_decision_threshold": "global_f1_tau099",
    "macro_game_f1_at_decision_threshold": "macro_game_f1_tau099",
    "worst_game_f1_at_decision_threshold": "worst_game_f1_tau099",
}


def _metric_value(metrics: dict, name: str) -> Any:
    """Read a metric by canonical name, falling back to its legacy alias."""
    value = metrics.get(name)
    if value is not None:
        return value
    alias = _LEGACY_METRIC_ALIASES.get(name)
    if alias is not None:
        return metrics.get(alias)
    return None


def _selection_mode(evaluation_config: dict) -> str:
    """Resolve the selection strategy, defaulting to the legacy behavior.

    When ``selection_mode`` is absent, a ``composite`` ``selection_metric``
    implies composite selection; anything else stays plain metric selection.
    """
    mode = evaluation_config.get("selection_mode")
    if mode is not None:
        return str(mode)
    if evaluation_config.get("selection_metric") == "composite":
        return "composite"
    return "metric"


def _selection_score(metrics: dict, evaluation_config: dict) -> tuple[float, bool]:
    metric_name = evaluation_config.get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    minimum_worst = evaluation_config.get("minimum_worst_game_f1")
    eligible = minimum_worst is None or float(
        _metric_value(metrics, "worst_game_f1_at_decision_threshold") or 0.0
    ) >= float(minimum_worst)
    if metric_name == "composite":
        weights = evaluation_config.get("selection_weights", {})
        components = {
            "global_f1": float(
                _metric_value(metrics, "global_f1_at_decision_threshold") or 0.0
            ),
            "macro_game_f1": float(
                _metric_value(metrics, "macro_game_f1_at_decision_threshold") or 0.0
            ),
            "worst_game_f1": float(
                _metric_value(metrics, "worst_game_f1_at_decision_threshold") or 0.0
            ),
        }
        total_weight = sum(float(weights.get(key, 0.0)) for key in components)
        if total_weight <= 0:
            raise ValueError(
                "evaluation.selection_weights must contain a positive weight"
            )
        score = (
            sum(components[key] * float(weights.get(key, 0.0)) for key in components)
            / total_weight
        )
    else:
        value = _metric_value(metrics, metric_name)
        if value is None:
            raise KeyError(
                f"Selection metric is absent from evaluation output: {metric_name}"
            )
        score = float(value)
    return score, eligible


def _selection_eligible(metrics: dict, evaluation_config: dict) -> bool:
    """Return whether ``metrics`` passes the configured selection gates.

    metric/composite: the existing ``minimum_worst_game_f1`` gate.

    constrained: every non-null FPR/recall gate must pass
    (``global_fpr <= max_global_fpr``, ``worst_game_fpr <=
    max_worst_game_fpr``, ``global_positive_recall >= min_positive_recall``),
    with ``minimum_worst_game_f1`` still applied as an extra gate when set.
    """
    if _selection_mode(evaluation_config) == "constrained":
        eligible = True
        max_global_fpr = evaluation_config.get("max_global_fpr")
        if eligible and max_global_fpr is not None:
            global_fpr = _metric_value(metrics, "global_fpr_at_decision_threshold")
            eligible = global_fpr is not None and float(global_fpr) <= float(
                max_global_fpr
            )
        max_worst_game_fpr = evaluation_config.get("max_worst_game_fpr")
        if eligible and max_worst_game_fpr is not None:
            worst_game_fpr = _metric_value(
                metrics, "worst_game_fpr_at_decision_threshold"
            )
            eligible = worst_game_fpr is not None and float(worst_game_fpr) <= float(
                max_worst_game_fpr
            )
        min_positive_recall = evaluation_config.get("min_positive_recall")
        if eligible and min_positive_recall is not None:
            recall = _metric_value(
                metrics, "global_positive_recall_at_decision_threshold"
            )
            eligible = recall is not None and float(recall) >= float(
                min_positive_recall
            )
        max_worst_subtype_fpr = evaluation_config.get("max_worst_subtype_fpr")
        if eligible and max_worst_subtype_fpr is not None:
            worst_subtype_fpr = _metric_value(
                metrics, "worst_subtype_fpr_at_decision_threshold"
            )
            eligible = worst_subtype_fpr is not None and float(
                worst_subtype_fpr
            ) <= float(max_worst_subtype_fpr)
        minimum_worst = evaluation_config.get("minimum_worst_game_f1")
        if eligible and minimum_worst is not None:
            worst_f1 = float(
                _metric_value(metrics, "worst_game_f1_at_decision_threshold") or 0.0
            )
            eligible = worst_f1 >= float(minimum_worst)
        return eligible
    # metric / composite: keep the existing minimum_worst_game_f1 gate.
    return bool(_selection_score(metrics, evaluation_config)[1])


def _selection_rank_key(
    metrics: dict, evaluation_config: dict
) -> float | tuple[float, ...]:
    """Return the ordering key used to compare selection candidates.

    metric/composite: the existing scalar selection score (float).

    constrained: ``(global_positive_recall, worst_game_positive_recall,
    -negative_score_p999)``; tuples compare lexicographically so selection
    maximizes recall, then worst-game recall, then minimizes negative
    score p99.9.
    """
    if _selection_mode(evaluation_config) == "constrained":
        return (
            float(
                _metric_value(
                    metrics,
                    "global_positive_recall_at_decision_threshold",
                )
                or 0.0
            ),
            float(
                _metric_value(
                    metrics,
                    "worst_game_positive_recall_at_decision_threshold",
                )
                or 0.0
            ),
            -float(_metric_value(metrics, "negative_score_p999") or 0.0),
        )
    return float(_selection_score(metrics, evaluation_config)[0])


def _annotate_selection(metrics: dict, evaluation_config: dict) -> dict:
    metrics["selection_mode"] = _selection_mode(evaluation_config)
    metrics["selection_eligible"] = _selection_eligible(metrics, evaluation_config)
    if _selection_mode(evaluation_config) == "constrained":
        primary = _metric_value(metrics, "global_positive_recall_at_decision_threshold")
        metrics["selection_score"] = float(primary or 0.0)
        metrics["selection_metric"] = "global_positive_recall_at_decision_threshold"
        metrics["selection_constraints"] = {
            "global_fpr": _metric_value(metrics, "global_fpr_at_decision_threshold"),
            "worst_game_fpr": _metric_value(
                metrics, "worst_game_fpr_at_decision_threshold"
            ),
            "global_positive_recall": _metric_value(
                metrics, "global_positive_recall_at_decision_threshold"
            ),
            "max_global_fpr": evaluation_config.get("max_global_fpr"),
            "max_worst_game_fpr": evaluation_config.get("max_worst_game_fpr"),
            "min_positive_recall": evaluation_config.get("min_positive_recall"),
            "negative_score_p999": _metric_value(metrics, "negative_score_p999"),
        }
        return metrics
    score, eligible = _selection_score(metrics, evaluation_config)
    metrics["selection_metric"] = evaluation_config.get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    metrics["selection_score"] = score
    metrics["selection_eligible"] = eligible
    metrics["minimum_worst_game_f1"] = evaluation_config.get("minimum_worst_game_f1")
    return metrics


def _is_better_model(candidate: dict, incumbent: dict, evaluation_config: dict) -> bool:
    """Return whether ``candidate`` beats ``incumbent`` under selection.

    The candidate must be eligible; an empty incumbent is always beaten.
    Otherwise the two rank keys are compared (scalar float for
    metric/composite, lexicographic tuple for constrained), so both keys
    share a type for a fixed mode.
    """
    if not _selection_eligible(candidate, evaluation_config):
        return False
    if not incumbent:
        return True
    if not _selection_eligible(incumbent, evaluation_config):
        return True
    candidate_key = _selection_rank_key(candidate, evaluation_config)
    incumbent_key = _selection_rank_key(incumbent, evaluation_config)
    # For a fixed selection mode both keys share a type (float or a tuple);
    # mypy cannot know that, so the comparison is dynamically safe.
    return candidate_key > incumbent_key  # type: ignore[operator]


def _save_best_enabled(checkpoint_config: dict) -> bool:
    return bool(
        checkpoint_config.get(
            "save_best_selection",
            checkpoint_config.get("save_best_test_f1", True),
        )
    )
