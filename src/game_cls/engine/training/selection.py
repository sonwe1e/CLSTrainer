from __future__ import annotations

from typing import Any


def _metric_value(metrics: dict, name: str) -> Any:
    """Read one canonical evaluator metric."""
    return metrics.get(name)


def _selection_mode(evaluation_config: dict) -> str:
    """Resolve the configured selection strategy."""
    return str(evaluation_config.get("selection_mode", "metric"))


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


def selection_sort_value(
    metrics: dict,
    evaluation_config: dict,
    *,
    include_eligibility: bool = True,
) -> list[float]:
    """Return the selection rank key as an orderable JSON-serializable list.

    Every consumer that has to order selection candidates (best checkpoint,
    early stopping, topk, the Run index) must go through this helper so the
    orderings can never disagree. The scalar rank key of metric/composite
    mode becomes a one-element list and the constrained tuple becomes a
    three-element list, so the value survives a round-trip through
    ``topk_registry.json`` / ``index.jsonl`` and still compares
    lexicographically (a JSON list decodes back to a list, a tuple would
    not).

    Bigger is always better, in every mode: the constrained key already
    stores ``-negative_score_p999`` so its "lower is better" component
    maximizes like the rest.

    With ``include_eligibility`` (the default) the key is prefixed with
    ``1.0``/``0.0``, which places every eligible candidate above every
    ineligible one -- the same precedence ``_is_better_model`` applies. Pass
    ``include_eligibility=False`` when the caller gates on eligibility
    itself and needs the bare metric key (early stopping, where the prefix
    would otherwise absorb ``min_delta``).
    """
    key = _selection_rank_key(metrics, evaluation_config)
    parts = (
        [float(component) for component in key]
        if isinstance(key, tuple)
        else [float(key)]
    )
    if include_eligibility:
        return [
            1.0 if _selection_eligible(metrics, evaluation_config) else 0.0,
            *parts,
        ]
    return parts


def selection_report_fields(metrics: dict, evaluation_config: dict) -> dict:
    """Return the selection facts a Run has to publish for comparison.

    ``selection_score`` alone cannot explain a constrained decision: two runs
    can share a recall and still differ on eligibility, worst-game recall or
    the negative tail. The Run index and ``run compare`` both read these
    fields, so they live next to the selection rules instead of being spelled
    out twice.
    """
    # The bare key's leading component is the primary objective in every
    # mode (the scalar score, or global positive recall when constrained).
    primary = selection_sort_value(
        metrics, evaluation_config, include_eligibility=False
    )[0]
    return {
        "selection_mode": _selection_mode(evaluation_config),
        "selection_eligible": _selection_eligible(metrics, evaluation_config),
        "selection_sort_value": selection_sort_value(metrics, evaluation_config),
        "selection_score": primary,
        "global_positive_recall": _metric_value(
            metrics, "global_positive_recall_at_decision_threshold"
        ),
        "worst_game_positive_recall": _metric_value(
            metrics, "worst_game_positive_recall_at_decision_threshold"
        ),
        "negative_score_p999": _metric_value(metrics, "negative_score_p999"),
        "global_fpr": _metric_value(metrics, "global_fpr_at_decision_threshold"),
        "worst_game_fpr": _metric_value(
            metrics, "worst_game_fpr_at_decision_threshold"
        ),
    }


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
    return bool(checkpoint_config.get("save_best_selection", True))
