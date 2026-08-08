from __future__ import annotations

from game_cls.engine.training.selection import (
    _selection_eligible,
    selection_sort_value,
)

# ``monitor`` values that mean "follow the selection contract" instead of
# naming a single numeric metric. Everything else (cross_entropy,
# objective_loss, worst_game_f1_*) keeps the plain float comparison.
_SELECTION_MONITORS = frozenset({"selection_score"})

# _selection_verdict outcomes.
_ABSTAIN = "abstain"  # monitored value unavailable: patience must not move
_PLATEAU = "plateau"  # a real evaluation that did not improve


def _early_stopping_defaults() -> dict:
    return {
        "best_value": None,
        "best_step": None,
        "bad_evaluation_count": 0,
        "stop_reason": None,
        "stopped_at_step": None,
    }


def _follows_selection_contract(early_config: dict) -> bool:
    """Whether ``monitor`` delegates to the selection rank key."""
    return str(early_config.get("monitor", "selection_score")) in _SELECTION_MONITORS


def _early_stopping_monitor_label(early_config: dict) -> str:
    """Human-readable monitor name for logs and stop reasons.

    The selection monitor is not a single metric, so printing
    ``monitor=selection_score`` in constrained mode would name a scalar the
    decision never looked at.
    """
    monitor = str(early_config.get("monitor", "selection_score"))
    if _follows_selection_contract(early_config):
        return "selection_rank_key"
    return monitor


def _persisted_key(best_value: object) -> list[float] | None:
    """Normalize a persisted ``best_value`` into a comparable key list.

    Return a comparable list for the persisted rank key.
    """
    if isinstance(best_value, (int, float)) and not isinstance(best_value, bool):
        return [float(best_value)]
    if isinstance(best_value, (list, tuple)):
        try:
            return [float(component) for component in best_value]
        except (TypeError, ValueError):
            return None
    return None


def _selection_improved(key: list[float], best_value: object, min_delta: float) -> bool:
    """Whether ``key`` beats ``best_value`` under the selection contract.

    ``min_delta`` applies to the primary component only -- that is the
    objective being maximized, and the remaining components are
    tie-breakers, not quantities a user can pick a threshold for. A
    candidate therefore improves when the primary component gains more than
    ``min_delta``, or when it stays inside the +/-``min_delta`` dead-band and
    a later tie-breaker strictly improves. Without the second clause a
    positive ``min_delta`` would mute the tie-breakers entirely, so a model
    with the same recall but a better worst-game recall would read as a
    plateau while best-checkpoint selection called it an improvement.
    """
    best_key = _persisted_key(best_value)
    if best_key is None:
        return True
    if key[0] > best_key[0] + min_delta:
        return True
    return abs(key[0] - best_key[0]) <= min_delta and key[1:] > best_key[1:]


def _selection_verdict(
    early_config: dict, metrics: dict, evaluation_config: dict, best_value: object
) -> list[float] | str:
    """Selection-contract verdict: the new best key, ``_PLATEAU`` or ``_ABSTAIN``."""
    if not metrics:
        return _ABSTAIN
    try:
        eligible = _selection_eligible(metrics, evaluation_config)
        key = selection_sort_value(
            metrics, evaluation_config, include_eligibility=False
        )
    except (KeyError, TypeError, ValueError):
        # The selection metric is absent from this payload (e.g. a partial or
        # mocked evaluation): abstain rather than burn patience.
        return _ABSTAIN
    if not eligible:
        # An ineligible model is not a selection candidate at all, so it can
        # never be an improvement -- but it is a real non-improving
        # evaluation and must count toward patience.
        return _PLATEAU
    min_delta = float(early_config.get("min_delta", 0.0))
    if _selection_improved(key, best_value, min_delta):
        return key
    return _PLATEAU


def _update_early_stopping(
    state: dict,
    early_config: dict,
    metrics: dict,
    global_step: int,
    evaluation_config: dict | None = None,
) -> bool:
    """Update patience state after a full validation; returns stop decision.

    Two monitor contracts share this function:

    * ``monitor: selection_score`` (the default) follows the *selection*
      contract: improvement is judged by ``selection_sort_value``, the same
      key that picks the best checkpoint, and an ineligible evaluation can
      never be an improvement -- it increments patience instead of resetting
      it. ``mode`` is ignored here because the rank key is built so that
      bigger is always better. This needs ``evaluation_config``; without it
      the numeric path is used, which keeps standalone helper callers working.
    * any other ``monitor`` names one numeric metric and keeps the original
      ``mode: max|min`` float comparison, byte for byte.

    Both paths abstain (no state change, no stop) when the monitored value is
    missing, so a payload from a skipped or partial evaluation cannot consume
    patience.
    """
    if evaluation_config is not None and _follows_selection_contract(early_config):
        verdict = _selection_verdict(
            early_config, metrics, evaluation_config, state.get("best_value")
        )
        if verdict == _ABSTAIN:
            return False
        improved = verdict != _PLATEAU
        if improved:
            state["best_value"] = verdict
    else:
        monitor = str(early_config.get("monitor", "selection_score"))
        value = metrics.get(monitor)
        if not isinstance(value, (int, float)):
            return False
        value = float(value)
        mode = early_config.get("mode", "max")
        min_delta = float(early_config.get("min_delta", 0.0))
        best_value = state.get("best_value")
        # A run resumed after a monitor change can hold a rank-key list here;
        # it is not comparable to a float, so treat it as no incumbent.
        if not isinstance(best_value, (int, float)) or isinstance(best_value, bool):
            best_value = None
        if best_value is None:
            improved = True
        elif mode == "max":
            improved = value > float(best_value) + min_delta
        else:
            improved = value < float(best_value) - min_delta
        if improved:
            state["best_value"] = value
    if improved:
        state["best_step"] = int(global_step)
        state["bad_evaluation_count"] = 0
        return False
    burn_in = int(early_config.get("burn_in_steps", 0))
    if global_step < burn_in:
        # Burn-in: an unimproving evaluation must not consume patience. The
        # ``improved`` branch above still advances ``best_step``/``best_value``
        # inside burn-in, but a run inside burn-in can never stop early. Before
        # this gate, bad evaluations inside burn-in were counted, so a run with
        # val_full_every_steps=2000, burn_in_steps=6000, patience=3 stopped the
        # instant it crossed burn-in instead of giving the schedule room to
        # warm up (the audit P1-1).
        return False
    state["bad_evaluation_count"] = int(state.get("bad_evaluation_count", 0)) + 1
    patience = int(early_config.get("patience_evaluations", 3))
    return state["bad_evaluation_count"] >= patience
