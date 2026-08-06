from __future__ import annotations


def _early_stopping_defaults() -> dict:
    return {
        "best_value": None,
        "best_step": None,
        "bad_evaluation_count": 0,
        "stop_reason": None,
        "stopped_at_step": None,
    }


def _update_early_stopping(
    state: dict, early_config: dict, metrics: dict, global_step: int
) -> bool:
    """Update patience state after a full validation; returns stop decision."""
    monitor = early_config.get("monitor", "selection_score")
    value = metrics.get(monitor)
    if not isinstance(value, (int, float)):
        return False
    value = float(value)
    mode = early_config.get("mode", "max")
    min_delta = float(early_config.get("min_delta", 0.0))
    best_value = state.get("best_value")
    if best_value is None:
        improved = True
    elif mode == "max":
        improved = value > float(best_value) + min_delta
    else:
        improved = value < float(best_value) - min_delta
    if improved:
        state["best_value"] = value
        state["best_step"] = int(global_step)
        state["bad_evaluation_count"] = 0
        return False
    state["bad_evaluation_count"] = int(state.get("bad_evaluation_count", 0)) + 1
    patience = int(early_config.get("patience_evaluations", 3))
    burn_in = int(early_config.get("burn_in_steps", 0))
    return state["bad_evaluation_count"] >= patience and global_step >= burn_in
