from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from game_cls.engine.checkpoint import (
    clone_checkpoint_pair,
    remove_checkpoint_pair,
)
from game_cls.engine.training.selection import (
    _metric_value,
    _selection_eligible,
    selection_report_fields,
    selection_sort_value,
)
from game_cls.runs import (
    STATE_FAILED,
    STATE_SUCCEEDED,
    append_run_index,
    read_manifest,
    render_overview_html,
    render_summary_md,
    update_status,
    write_manifest,
)

# ``topk_monitor`` values that mean "rank by the selection contract".
_SELECTION_TOPK_MONITORS = frozenset({"selection_score", "selection"})


def _topk_entry_sort_value(entry: dict, *, lower_better: bool) -> list[float]:
    """Sort key of an existing registry entry, best-first when descending.

    Entries written by the current code carry ``sort_value``. Entries restored
    from a checkpoint written before that only have the scalar ``value``, so
    it is lifted into the same shape (negated for lower-is-better monitors,
    which the rank key never needs because it always maximizes).
    """
    sort_value = entry.get("sort_value")
    if isinstance(sort_value, (list, tuple)):
        return [float(component) for component in sort_value]
    value = float(entry.get("value") or 0.0)
    return [-value] if lower_better else [value]


def _maybe_save_topk(
    *,
    output_dir: Path,
    checkpoint_cfg: dict,
    metrics: dict,
    global_step: int,
    evaluation_state: dict,
    evaluation_cfg: dict | None = None,
) -> None:
    """Register the current full-validation checkpoint in the topk list.

    Clones the just-written ``last`` pair to ``model_topk_<step>.pth``,
    evicts the worst entries beyond ``save_topk``, and persists the
    registry both in ``evaluation_state`` (so resume continues the list)
    and in ``checkpoints/topk_registry.json``.

    ``topk_monitor: selection_score`` (the default) ranks by
    ``selection_sort_value``, so the topk order is the order
    ``_is_better_model`` would produce -- in constrained mode that is
    recall -> worst-game recall -> negative p99.9, not a single scalar. Any
    other monitor names one numeric metric and keeps the original scalar
    ordering (lower is better for ``cross_entropy``).

    Ineligible checkpoints are **admitted but ranked strictly below every
    eligible one** (eligibility is the leading component of the sort key)
    rather than filtered out. Dropping them would leave the registry empty
    for the whole early phase of a constrained run, where no candidate meets
    the FPR gates yet, and those near-miss snapshots are exactly what an
    operator inspects; ranking them last still guarantees an eligible
    checkpoint is never evicted in favor of an ineligible one.
    """
    topk = int(checkpoint_cfg.get("save_topk", 0))
    if topk <= 0:
        return
    monitor = str(checkpoint_cfg.get("topk_monitor", "selection_score"))
    by_selection = evaluation_cfg is not None and monitor in _SELECTION_TOPK_MONITORS
    lower_better = not by_selection and monitor == "cross_entropy"
    eligible = True
    if by_selection:
        assert evaluation_cfg is not None  # narrowed by by_selection
        try:
            sort_value = selection_sort_value(metrics, evaluation_cfg)
            eligible = _selection_eligible(metrics, evaluation_cfg)
        except (KeyError, TypeError, ValueError):
            # The selection metric is absent from this payload; registering an
            # unrankable checkpoint would corrupt the order.
            return
        # Reported value: the primary objective, so `run show` / summary.md
        # keep printing one comparable number per entry.
        value = sort_value[1]
    else:
        raw = _metric_value(metrics, monitor)
        if not isinstance(raw, (int, float)):
            return
        value = float(raw)
        sort_value = [-value] if lower_better else [value]
    registry = list(evaluation_state.get("topk_registry") or [])
    if any(int(entry.get("step", -1)) == int(global_step) for entry in registry):
        return
    if len(registry) >= topk and not any(
        _topk_entry_sort_value(entry, lower_better=lower_better) <= sort_value
        for entry in registry
    ):
        return
    tag = f"topk_{global_step:08d}"
    clone_checkpoint_pair(output_dir / "checkpoints", "last", tag)
    registry.append(
        {
            "step": int(global_step),
            "value": value,
            "sort_value": sort_value,
            "eligible": bool(eligible),
            "monitor": monitor,
            "tag": tag,
            "filename": f"model_{tag}.pth",
        }
    )
    registry.sort(
        key=lambda entry: _topk_entry_sort_value(entry, lower_better=lower_better),
        reverse=True,
    )
    evicted = registry[topk:]
    registry = registry[:topk]
    for entry in evicted:
        remove_checkpoint_pair(output_dir / "checkpoints", entry["tag"])
    evaluation_state["topk_registry"] = registry
    try:
        from game_cls.engine.checkpoint import _atomic_json_save

        _atomic_json_save(
            {
                "monitor": monitor,
                "topk": topk,
                "entries": registry,
            },
            output_dir / "checkpoints" / "topk_registry.json",
        )
    except OSError:
        pass


def _read_status(output_dir: Path) -> dict:
    """Load status.json without failing when it is absent or corrupt."""
    status_path = output_dir / "status.json"
    if not status_path.is_file():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _iso_now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_run_manifest(
    output_dir: Path,
    *,
    config: dict,
    run_meta: dict | None,
    run_id: str | None,
    run_mode: str,
    world_size: int,
) -> str | None:
    """Create or preserve the run's immutable manifest.

    The manifest is written exactly once. If it already exists (resume or a
    re-run against the same directory) it is left untouched and the original
    ``run_id`` is returned so status/index records stay attached to the same
    Run identity.
    """
    existing = read_manifest(output_dir)
    if existing is not None:
        return existing.get("run_id")
    meta = run_meta or {}
    manifest = {
        "run_id": run_id,
        "run_mode": run_mode,
        "run_name": config["experiment"].get("name"),
        "created": _iso_now(),
        "command": meta.get("command"),
        "config_file": meta.get("config_file"),
        "seed": config["experiment"].get("seed"),
        "world_size": world_size,
        "accelerator": config["device"].get("accelerator"),
        "decision_threshold": config.get("decision", {}).get("threshold"),
        "resume_checkpoint": config["train"].get("resume_path"),
        "resumed_from": meta.get("resumed_from"),
        "forked_from": meta.get("forked_from"),
        "parent_run_id": meta.get("parent_run_id"),
        "base_checkpoint": config["model"].get("checkpoint_path"),
        "base_checkpoint_sha256": meta.get("base_checkpoint_sha256"),
        "environment": meta.get("environment"),
    }
    write_manifest(output_dir, manifest)
    return run_id


def _write_resolved_config(output_dir: Path, config: dict) -> None:
    """Write the resolved config snapshot.

    ``resolved_config.initial.json`` captures the very first resolved
    configuration of the run and is never touched again; every resume
    overwrites only ``resolved_config.json`` with the new snapshot.
    """
    resolved_path = output_dir / "resolved_config.json"
    initial_path = output_dir / "resolved_config.initial.json"
    if not initial_path.exists():
        if resolved_path.exists():
            shutil.copy2(resolved_path, initial_path)
        else:
            resolved_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shutil.copy2(resolved_path, initial_path)
            return
    resolved_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _record_run_failure(
    output_dir: Path,
    exc: BaseException,
    *,
    global_step: int,
    run_id: str | None,
    run_mode: str,
    runs_root: Path,
    config: dict,
    started_wall: float,
) -> None:
    import traceback

    failure_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    try:
        (output_dir / "failure.log").write_text(failure_text, encoding="utf-8")
        update_status(
            output_dir,
            state=STATE_FAILED,
            step=global_step,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            traceback_file="failure.log",
            finished=_iso_now(),
        )
        append_run_index(
            runs_root,
            {
                "run_id": run_id,
                "name": config["experiment"].get("name"),
                "output_dir": str(output_dir),
                "state": STATE_FAILED,
                "error_type": type(exc).__name__,
                "global_step": global_step,
                "finished": _iso_now(),
                "duration_seconds": round(time.time() - started_wall, 3),
            },
        )
    except OSError:
        pass


def _finalize_run_success(
    *,
    output_dir: Path,
    config: dict,
    run_id: str | None,
    run_mode: str,
    runs_root: Path,
    summary_payload: dict,
    global_step: int,
    total_steps: int,
    best_metrics: dict,
    started_wall: float,
    parent_run_id: str | None = None,
) -> None:
    from datetime import datetime

    finished = _iso_now()
    started_iso = (
        datetime.fromtimestamp(started_wall).astimezone().isoformat(timespec="seconds")
    )
    duration = time.time() - started_wall
    update_status(
        output_dir,
        state=STATE_SUCCEEDED,
        step=global_step,
        finished=finished,
    )
    try:
        summary_md = render_summary_md(
            run_id=run_id,
            run_dir=output_dir,
            state=STATE_SUCCEEDED,
            started=started_iso,
            finished=finished,
            duration_seconds=duration,
            config=config,
            summary_payload=summary_payload,
        )
        (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    except OSError:
        pass
    try:
        overview_html = render_overview_html(
            run_dir=output_dir,
            run_id=run_id,
            state=STATE_SUCCEEDED,
            started=started_iso,
            finished=finished,
            duration_seconds=duration,
            config=config,
            summary_payload=summary_payload,
        )
        (output_dir / "overview.html").write_text(overview_html, encoding="utf-8")
    except OSError:
        pass
    if run_mode == "unique" or run_id is not None:
        selection = _run_index_selection(best_metrics, config.get("evaluation") or {})
        append_run_index(
            runs_root,
            {
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "name": config["experiment"].get("name"),
                "output_dir": str(output_dir),
                "state": STATE_SUCCEEDED,
                "started": started_iso,
                "finished": finished,
                "duration_seconds": round(duration, 3),
                "global_step": global_step,
                "total_steps": total_steps,
                # Kept flat for readers that predate the block below.
                "selection_score": selection.get("selection_score"),
                "selection": selection,
            },
        )


def _run_index_selection(best_metrics: dict, evaluation_cfg: dict) -> dict:
    """Selection block appended to the Run index for ``run compare``.

    A lone ``selection_score`` cannot explain a constrained decision, so the
    index also carries eligibility, the full rank key and the metrics the
    gates and tie-breakers read. Recording it must never fail a finished run:
    metrics that do not match the current selection config (an old resumed
    checkpoint, a partial payload) degrade to the raw ``selection_score``.
    """
    if not isinstance(best_metrics, dict) or not best_metrics:
        return {"selection_score": None}
    try:
        return selection_report_fields(best_metrics, evaluation_cfg)
    except (KeyError, TypeError, ValueError):
        return {"selection_score": best_metrics.get("selection_score")}


# Evaluation kinds and their train/validation/test protocol roles.
_EVALUATION_ROLES: dict[str, str] = {
    "train_probe": "train_probe",
    "val_quick": "validation",
    "val_full": "validation",
    "val_full_final": "validation",
    "test_full": "test",
}


_EVALUATION_SCOPES: dict[str, str] = {
    "train_probe": "probe",
    "val_quick": "quick",
    "val_full": "full",
    "val_full_final": "final",
    "test_full": "full",
}


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def _append_training_metrics(path: Path, payload: dict) -> None:
    _append_jsonl(path, payload)


def _append_evaluation_history(output_dir: Path, record: dict) -> None:
    """Append one evaluation to <run>/metrics/evaluation.jsonl."""
    path = output_dir / "metrics" / "evaluation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(path, record)


def _evaluation_history_record(
    metrics: dict, *, kind: str, global_step: int, extra: dict | None = None
) -> dict:
    record = {
        "step": int(global_step),
        "split": _EVALUATION_ROLES[kind],
        "scope": _EVALUATION_SCOPES[kind],
        "kind": kind,
        "cross_entropy": metrics.get("cross_entropy"),
        "threshold_loss": metrics.get("threshold_loss"),
        "objective_loss": metrics.get("objective_loss"),
        "brier_score": metrics.get("brier_score"),
        "ece_20_bins": metrics.get("ece_20_bins"),
        "global_f1_at_decision_threshold": metrics.get(
            "global_f1_at_decision_threshold"
        ),
        "macro_game_f1_at_decision_threshold": metrics.get(
            "macro_game_f1_at_decision_threshold"
        ),
        "worst_game_f1_at_decision_threshold": metrics.get(
            "worst_game_f1_at_decision_threshold"
        ),
        "global_f1_tau099": metrics.get("global_f1_tau099"),
        "macro_game_f1_tau099": metrics.get("macro_game_f1_tau099"),
        "worst_game_f1_tau099": metrics.get("worst_game_f1_tau099"),
        "selection_score": metrics.get("selection_score"),
        "positive_margin_pass_rate": metrics.get("positive_margin_pass_rate"),
        "negative_margin_pass_rate": metrics.get("negative_margin_pass_rate"),
        "positive_margin_p10": metrics.get("positive_margin_p10"),
        "positive_margin_p50": metrics.get("positive_margin_p50"),
        "positive_margin_p90": metrics.get("positive_margin_p90"),
        "negative_margin_p10": metrics.get("negative_margin_p10"),
        "negative_margin_p50": metrics.get("negative_margin_p50"),
        "negative_margin_p90": metrics.get("negative_margin_p90"),
        "sample_count": metrics.get("sample_count"),
    }
    if extra:
        record.update(extra)
    return record
