from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from game_cls.config_schema import finalize_config, split_role_warnings
from game_cls.data.collate import pair_collate
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.checkpoint import (
    capture_random_state,
    clone_checkpoint_pair,
    remove_checkpoint_pair,
    restore_random_state,
    restore_training_checkpoint,
    save_checkpoint_pair,
    unwrap_model,
)
from game_cls.engine.device import autocast_context
from game_cls.engine.distributed import (
    cleanup_distributed,
    distributed_barrier,
    initialize_runtime,
    is_distributed,
)
from game_cls.engine.evaluator import EvaluationOutput, evaluate
from game_cls.losses.threshold_loss import (
    combined_loss,
    probability_threshold_to_margin,
    threshold_weight_at_step,
    threshold_weight_from_steps,
)
from game_cls.model.builder import build_model
from game_cls.model.checkpoint_loader import (
    load_model_checkpoint,
    validate_production_load,
)
from game_cls.model.freeze_policy import (
    assert_frozen_parameters_unchanged,
    configure_trainable_parameters,
    set_frozen_backbone_train_mode,
    snapshot_frozen_parameters,
)
from game_cls.reports.error_writer import (
    prepare_evaluation_directory,
    write_evaluation_report,
)
from game_cls.runs import (
    STATE_FAILED,
    STATE_RUNNING,
    STATE_SUCCEEDED,
    allocate_run_dir,
    append_resume_event,
    append_run_index,
    atomic_write_text,
    read_manifest,
    render_overview_html,
    render_summary_md,
    update_status,
    write_manifest,
)

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


def _maybe_save_topk(
    *,
    output_dir: Path,
    checkpoint_cfg: dict,
    metrics: dict,
    global_step: int,
    evaluation_state: dict,
) -> None:
    """Register the current full-validation checkpoint in the topk list.

    Clones the just-written ``last`` pair to ``model_topk_<step>.pth``,
    evicts the worst entries beyond ``save_topk``, and persists the
    registry both in ``evaluation_state`` (so resume continues the list)
    and in ``checkpoints/topk_registry.json``.
    """
    topk = int(checkpoint_cfg.get("save_topk", 0))
    if topk <= 0:
        return
    monitor = str(checkpoint_cfg.get("topk_monitor", "selection_score"))
    value = _metric_value(metrics, monitor)
    if not isinstance(value, (int, float)):
        return
    value = float(value)
    lower_better = monitor == "cross_entropy"
    registry = list(evaluation_state.get("topk_registry") or [])
    if any(int(entry.get("step", -1)) == int(global_step) for entry in registry):
        return
    worse_than_new = (
        (lambda entry: entry["value"] >= value)
        if lower_better
        else (lambda entry: entry["value"] <= value)
    )
    if len(registry) >= topk and not any(worse_than_new(entry) for entry in registry):
        return
    tag = f"topk_{global_step:08d}"
    clone_checkpoint_pair(output_dir / "checkpoints", "last", tag)
    registry.append(
        {
            "step": int(global_step),
            "value": value,
            "monitor": monitor,
            "tag": tag,
            "filename": f"model_{tag}.pth",
        }
    )
    registry.sort(key=lambda entry: entry["value"], reverse=not lower_better)
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
        (output_dir / "failure.log").write_text(
            failure_text, encoding="utf-8"
        )
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
    started_iso = datetime.fromtimestamp(started_wall).astimezone().isoformat(
        timespec="seconds"
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
        (output_dir / "overview.html").write_text(
            overview_html, encoding="utf-8"
        )
    except OSError:
        pass
    if run_mode == "unique" or run_id is not None:
        selection_score = (
            best_metrics.get("selection_score")
            if isinstance(best_metrics, dict)
            else None
        )
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
                "selection_score": selection_score,
            },
        )


class SyntheticPairDataset:
    def __init__(
        self, length: int, image_spec: ImageSpec, seed: int
    ) -> None:
        self.length = length
        self.image_spec = image_spec
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        import torch

        generator = torch.Generator().manual_seed(self.seed + index)
        label = index % 2
        images = torch.randint(
            0,
            96,
            (2, *self.image_spec.chw),
            dtype=torch.uint8,
            generator=generator,
        )
        if label:
            images[:, 0, : self.image_spec.height // 2] += 128
        return {
            "images": images,
            "label": label,
            "meta": {
                "game": f"synthetic_{index % 2}",
                "video_id": f"{index % 4:02d}",
                "frame0_id": index,
                "frame1_id": index + 2,
                "delta": 2,
                "image0_path": "",
                "image1_path": "",
            },
        }


@dataclass
class LoaderBundle:
    """DataLoader roles of the train/validation/test protocol.

    ============ =========================== ======= =======================
    role         source                      augment purpose
    ============ =========================== ======= =======================
    train        train split                 on      gradient updates
    train_probe  fixed train subset          off     generalization gap
    val_quick    fixed validation subset     off     high-frequency trends
    val_full     full validation split       off     selection + early stop
    test_full    full test split             off     only via ``evaluate``
    ============ =========================== ======= =======================

    ``test_full`` is ``None`` when the test split is aliased as validation;
    it never runs inside the training loop.
    """

    train: Any
    train_probe: Any
    val_quick: Any
    val_full: Any
    test_full: Any
    sampler: Any
    data_summary: dict


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


def has_independent_test(config: dict) -> bool:
    """True when test is a real holdout, not validation in disguise."""
    migration = (config.get("data") or {}).get("split_migration") or {}
    return not bool(migration.get("test_used_as_validation", False))


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
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
        "positive_margin_pass_rate": metrics.get(
            "positive_margin_pass_rate"
        ),
        "negative_margin_pass_rate": metrics.get(
            "negative_margin_pass_rate"
        ),
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


def _threshold_weight_for_eval(
    config: dict, step: int, total_steps: int
) -> float:
    """Current margin-loss weight, mirrored from the training schedule."""
    loss_cfg = config["loss"]
    max_weight = loss_cfg.get("threshold_loss_weight", 0.20)
    warmup_steps = loss_cfg.get("threshold_warmup_steps")
    ramp_steps = loss_cfg.get("threshold_ramp_steps")
    if warmup_steps is not None or ramp_steps is not None:
        return threshold_weight_from_steps(
            step, int(warmup_steps or 0), int(ramp_steps or 0), max_weight
        )
    return threshold_weight_at_step(
        step,
        total_steps,
        max_weight,
        loss_cfg.get("threshold_warmup_ratio", 0.10),
        loss_cfg.get("threshold_ramp_ratio", 0.20),
    )


def _new_interval_accumulator(device) -> dict:
    """Sample-weighted interval accumulators (device tensors, no sync)."""
    import torch

    return {
        "loss_sum": torch.zeros((), dtype=torch.float64, device=device),
        "ce_sum": torch.zeros((), dtype=torch.float64, device=device),
        "threshold_sum": torch.zeros((), dtype=torch.float64, device=device),
        "threshold_weight_sum": 0.0,
        "counts": torch.zeros(4, dtype=torch.int64, device=device),
        "samples": 0,
    }


def _reduce_interval_accumulator(accum: dict, device) -> dict:
    """Distributed-reduce interval accumulators and derive mean metrics."""
    import torch

    floating = torch.stack(
        (
            accum["loss_sum"],
            accum["ce_sum"],
            accum["threshold_sum"],
        )
    ).to(torch.float64)
    samples = torch.tensor(
        [accum["samples"]], dtype=torch.float64, device=device
    )
    counts = accum["counts"].clone()
    if is_distributed():
        import torch.distributed as dist

        dist.all_reduce(floating)
        dist.all_reduce(samples)
        dist.all_reduce(counts)
    sample_count = float(samples.item())
    loss_sum, ce_sum, threshold_sum = floating.tolist()
    tp, fp, fn, tn = counts.tolist()
    positive_total = tp + fn
    negative_total = fp + tn
    denominator = max(sample_count, 1.0)
    return {
        "interval_loss": loss_sum / denominator,
        "interval_ce": ce_sum / denominator,
        "interval_threshold_loss": threshold_sum / denominator,
        "interval_threshold_weight": accum["threshold_weight_sum"]
        / denominator,
        "interval_accuracy": (tp + tn) / max(sample_count, 1.0)
        if sample_count
        else 0.0,
        "interval_positive_recall_tau099": (
            tp / positive_total if positive_total else None
        ),
        "interval_negative_specificity_tau099": (
            tn / negative_total if negative_total else None
        ),
        "interval_samples": int(sample_count),
    }


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
    state["bad_evaluation_count"] = int(
        state.get("bad_evaluation_count", 0)
    ) + 1
    patience = int(early_config.get("patience_evaluations", 3))
    burn_in = int(early_config.get("burn_in_steps", 0))
    return (
        state["bad_evaluation_count"] >= patience
        and global_step >= burn_in
    )


def _tb_write_scalars(writer, prefix: str, payload: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize_device_for_metrics(device) -> None:
    import torch

    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _initialize_data_worker(
    worker_id: int,
    *,
    num_threads: int,
) -> None:
    import torch

    # Keep worker-level CPU parallelism from multiplying across DataLoader
    # processes. This function must remain at module scope for spawn pickling.
    del worker_id
    torch.set_num_threads(max(1, int(num_threads)))


def _dataloader_option(
    config: dict,
    role: str,
    name: str,
    default: Any,
) -> Any:
    root = config["dataloader"]
    scoped = root.get(role, {})
    if not isinstance(scoped, dict):
        raise TypeError(f"dataloader.{role} must be a mapping")
    return scoped.get(name, root.get(name, default))


def _validate_dataloader_config(config: dict) -> None:
    accelerator = str(config["device"]["accelerator"])
    allowed_contexts = {"spawn", "fork", "forkserver"}

    for role in ("train", "eval"):
        workers = int(
            _dataloader_option(config, role, "num_workers", 0)
        )
        if workers < 0:
            raise ValueError(
                f"dataloader.{role}.num_workers must be non-negative"
            )
        if workers == 0:
            continue

        context = _dataloader_option(
            config,
            role,
            "multiprocessing_context",
            "spawn" if accelerator == "npu" else None,
        )
        if context is not None:
            context = str(context)
        if context is not None and context not in allowed_contexts:
            raise ValueError(
                "dataloader multiprocessing_context must be one of "
                f"{sorted(allowed_contexts)}, got {context!r}"
            )
        if accelerator == "npu" and context != "spawn":
            raise RuntimeError(
                f"NPU dataloader.{role} must use spawn, got {context!r}"
            )
        if float(
            _dataloader_option(
                config, role, "timeout_seconds", 180
            )
        ) <= 0:
            raise ValueError(
                f"dataloader.{role}.timeout_seconds must be positive"
            )
        if int(
            _dataloader_option(config, role, "prefetch_factor", 2)
        ) <= 0:
            raise ValueError(
                f"dataloader.{role}.prefetch_factor must be positive"
            )
        if int(
            _dataloader_option(
                config, role, "worker_num_threads", 1
            )
        ) <= 0:
            raise ValueError(
                f"dataloader.{role}.worker_num_threads must be positive"
            )


def validate_training_config(config: dict) -> None:
    _validate_dataloader_config(config)
    data_cfg = config["data"]
    ImageSpec.from_config(data_cfg)
    model_cfg = config["model"]
    evaluation_cfg = config["evaluation"]
    evaluation_amp_dtype = str(
        evaluation_cfg.get(
            "amp_dtype", config["device"].get("amp_dtype", "bfloat16")
        )
    )
    if evaluation_amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError(
            "evaluation.amp_dtype must be float16 or bfloat16"
        )
    if int(evaluation_cfg.get("parquet_row_group_size", 4096)) <= 0:
        raise ValueError(
            "evaluation.parquet_row_group_size must be positive"
        )
    selection_metric = evaluation_cfg.get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    supported_selection_metrics = {
        "global_f1_at_decision_threshold",
        "macro_game_f1_at_decision_threshold",
        "worst_game_f1_at_decision_threshold",
        "global_f1_tau099",
        "macro_game_f1_tau099",
        "worst_game_f1_tau099",
        "composite",
    }
    if selection_metric not in supported_selection_metrics:
        raise ValueError(
            f"Unsupported evaluation.selection_metric: {selection_metric}"
        )
    if selection_metric == "composite" and sum(
        float(value)
        for value in evaluation_cfg.get("selection_weights", {}).values()
    ) <= 0:
        raise ValueError(
            "Composite model selection requires positive selection weights"
        )
    if not data_cfg.get("synthetic", False):
        # Production acceptance gate: with every full-validation source
        # disabled there is no model selection at all; such a run would
        # silently train blind.
        full_every = int(
            evaluation_cfg.get("val_full_every_steps", 0)
        )
        quick_every = int(
            evaluation_cfg.get("val_quick_every_steps", 0)
        )
        full_at_end = bool(
            evaluation_cfg.get("val_full_at_end", True)
        )
        if full_every <= 0 and not full_at_end:
            raise RuntimeError(
                "Production training requires at least one full-validation "
                "source: set evaluation.val_full_every_steps > 0 or "
                "evaluation.val_full_at_end=true (best-checkpoint selection "
                "and early stopping need validation signals)."
            )
        if full_every <= 0 and quick_every <= 0 and not full_at_end:
            raise RuntimeError(
                "Production training has every evaluation disabled; "
                "enable at least a quick or full validation cadence."
            )
        factory = str(model_cfg.get("factory", ""))
        checkpoint_path = model_cfg.get("checkpoint_path")
        if not factory or factory.endswith(":build_demo_model"):
            raise RuntimeError(
                "Production training cannot use build_demo_model; "
                "set model.factory to the real model factory."
            )
        if "your_package" in factory or "REPLACE_ME" in factory:
            raise RuntimeError(
                "Production model.factory is still a placeholder."
            )
        if not checkpoint_path:
            raise RuntimeError(
                "Production training requires model.checkpoint_path."
            )
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Production checkpoint does not exist: {checkpoint_path}"
            )
        if data_cfg.get("backend", "png") == "packed_uint8":
            if int(data_cfg.get("packed_max_open_shards", 16)) <= 0:
                raise ValueError("data.packed_max_open_shards must be positive")
            for key in ("train_packed_index", "test_packed_index"):
                packed_index = data_cfg.get(key)
                if not packed_index or not Path(packed_index).is_file():
                    raise FileNotFoundError(
                        f"Packed backend requires existing data.{key}: "
                        f"{packed_index}"
                    )
            val_packed_index = data_cfg.get("val_packed_index")
            if val_packed_index and not Path(val_packed_index).is_file():
                raise FileNotFoundError(
                    f"Packed backend requires existing data.val_packed_index: "
                    f"{val_packed_index}"
                )
            for split in ("train", "val", "test"):
                packed_index = data_cfg.get(f"{split}_packed_index")
                if not packed_index:
                    continue
                packed_video_index = data_cfg.get(
                    f"{split}_packed_video_index"
                ) or str(
                    Path(packed_index).with_name("packed_video_entries.parquet")
                )
                if not Path(packed_video_index).is_file():
                    raise FileNotFoundError(
                        "Packed backend requires the integer video index: "
                        f"{packed_video_index}"
                    )
    if data_cfg.get(
        "require_independent_test", False
    ) and not has_independent_test(config):
        raise RuntimeError(
            "data.require_independent_test is enabled but no independent "
            "test split exists: add data.val_index/data.val_video_index so "
            "the test split is held out for a single final evaluation."
        )
    if (
        config.get("distributed", {}).get("enabled", False)
        and not model_cfg.get("freeze_cls_batchnorm_stats", True)
    ):
        raise RuntimeError(
            "Distributed training with trainable cls BatchNorm statistics "
            "requires SyncBatchNorm; keep freeze_cls_batchnorm_stats=true."
        )


def _set_train_mode(model, model_config: dict) -> None:
    legacy = model_config.get("freeze_batchnorm_stats")
    set_frozen_backbone_train_mode(
        model,
        model_config.get("trainable_name_contains", "cls"),
        legacy,
        freeze_backbone_batchnorm_stats=model_config.get(
            "freeze_backbone_batchnorm_stats", True
        ),
        freeze_cls_batchnorm_stats=model_config.get(
            "freeze_cls_batchnorm_stats", True
        ),
    )


def build_optimizer_parameter_groups(model, weight_decay: float) -> list[dict]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _loader_common(config: dict, role: str) -> dict:
    workers = int(_dataloader_option(config, role, "num_workers", 0))
    if workers < 0:
        raise ValueError(
            f"dataloader.{role}.num_workers must be non-negative"
        )
    common = {
        "num_workers": workers,
        "pin_memory": bool(
            _dataloader_option(config, role, "pin_memory", False)
        ),
        "collate_fn": pair_collate,
    }
    if workers == 0:
        return common

    accelerator = str(config["device"]["accelerator"])
    context = _dataloader_option(
        config, role, "multiprocessing_context", None
    )
    if context is None and accelerator == "npu":
        context = "spawn"
    if context is not None:
        context = str(context)
    allowed_contexts = {"spawn", "fork", "forkserver"}
    if context is not None and context not in allowed_contexts:
        raise ValueError(
            "dataloader multiprocessing_context must be one of "
            f"{sorted(allowed_contexts)}, got {context!r}"
        )
    if accelerator == "npu" and context != "spawn":
        raise RuntimeError(
            "NPU DataLoader with num_workers > 0 must use "
            "multiprocessing_context=spawn"
        )

    timeout = float(
        _dataloader_option(config, role, "timeout_seconds", 180)
    )
    if timeout <= 0:
        raise ValueError(
            f"dataloader.{role}.timeout_seconds must be positive "
            "when num_workers > 0"
        )
    prefetch_factor = int(
        _dataloader_option(config, role, "prefetch_factor", 2)
    )
    if prefetch_factor <= 0:
        raise ValueError(
            f"dataloader.{role}.prefetch_factor must be positive"
        )
    worker_num_threads = int(
        _dataloader_option(config, role, "worker_num_threads", 1)
    )
    if worker_num_threads <= 0:
        raise ValueError(
            f"dataloader.{role}.worker_num_threads must be positive"
        )

    common.update(
        {
            "persistent_workers": bool(
                _dataloader_option(
                    config,
                    role,
                    "persistent_workers",
                    role == "train",
                )
            ),
            "prefetch_factor": prefetch_factor,
            "timeout": timeout,
            "multiprocessing_context": context,
            "worker_init_fn": partial(
                _initialize_data_worker,
                num_threads=worker_num_threads,
            ),
        }
    )
    return common


def _build_real_data_components(
    config: dict, rank: int, world_size: int
) -> dict:
    """Shared split metadata for training and standalone evaluation.

    Returns train/val/test video entries plus per-split decoders. The test
    entry is ``None`` when the config aliases test as validation.
    """
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import (
        audit_warning_messages,
        validate_audit_file,
    )
    from game_cls.data.video_index import read_video_entries_parquet

    data_cfg = config["data"]
    image_spec = ImageSpec.from_config(data_cfg)
    if data_cfg.get("strict_audit", True):
        audit_path = data_cfg.get("audit_path")
        if not audit_path:
            audit_path = str(Path(data_cfg["train_index"]).parent / "audit.json")
        audit = validate_audit_file(
            audit_path,
            image_spec=image_spec,
            scan_policy=ScanPolicy.from_config(data_cfg),
            duplicate_policy=DuplicatePolicy.from_config(data_cfg),
            require_test_delta=int(config["pair"]["test_delta"]),
            require_content_hash=bool(
                data_cfg.get("require_content_hash_audit", False)
            ),
            require_unique_video_keys=bool(
                data_cfg.get(
                    "require_unique_video_keys_across_splits", False
                )
            ),
            minimum_pairs_per_game_label_delta={
                int(key): int(value)
                for key, value in data_cfg.get(
                    "minimum_pairs_per_game_label_delta", {}
                ).items()
            },
        )
        if rank == 0:
            for warning in audit_warning_messages(audit):
                print(f"[WARNING] {warning}", flush=True)
    delta_probability = {
        int(key): float(value)
        for key, value in config["pair"]["train_delta_probability"].items()
    }
    test_delta = int(config["pair"]["test_delta"])
    backend_name = data_cfg.get("backend", "png")
    independent_test = has_independent_test(config)

    def _video_index_path(split: str) -> str:
        # Literal key names keep every schema leaf addressable (the CI
        # consumer guard scans for the exact strings).
        explicit_keys = {
            "train": ("train_video_index", "train_packed_video_index"),
            "val": ("val_video_index", "val_packed_video_index"),
            "test": ("test_video_index", "test_packed_video_index"),
        }
        fallback_keys = {
            "train": ("train_index", "train_packed_index"),
            "val": ("val_index", "val_packed_index"),
            "test": ("test_index", "test_packed_index"),
        }
        packed = backend_name == "packed_uint8"
        explicit = data_cfg.get(explicit_keys[split][1 if packed else 0])
        if explicit:
            return explicit
        return data_cfg[fallback_keys[split][1 if packed else 0]]

    train_videos = read_video_entries_parquet(
        _video_index_path("train"), delta_probability.keys()
    )
    val_videos = read_video_entries_parquet(
        _video_index_path("val"), (test_delta,)
    )
    test_videos = (
        read_video_entries_parquet(_video_index_path("test"), (test_delta,))
        if independent_test
        else None
    )

    decoders: dict[str, Any] = {"train": None, "val": None, "test": None}
    if backend_name == "packed_uint8":
        from game_cls.data.packed_backend import PackedUint8Backend

        for split in ("train", "val", "test"):
            if split == "test" and test_videos is None:
                continue
            decoders[split] = PackedUint8Backend(
                data_cfg[f"{split}_packed_index"],
                image_spec=image_spec,
                max_open_shards=int(
                    data_cfg.get("packed_max_open_shards", 16)
                ),
            )
    elif backend_name != "png":
        raise ValueError(f"Unsupported data backend: {backend_name}")

    return {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "decoders": decoders,
        "test_delta": test_delta,
        "delta_probability": delta_probability,
        "backend_name": backend_name,
        "independent_test": independent_test,
    }


def build_eval_loader_for_split(
    config: dict,
    split: str,
    rank: int,
    world_size: int,
    *,
    max_pairs_per_video: int | None = None,
):
    """Standalone evaluation DataLoader for the validation or test split.

    Used by ``cls-trainer evaluate``; the test split is only available
    when the config carries an independent test set.
    """
    from torch.utils.data import DataLoader

    from game_cls.data.lazy_pair_dataset import build_eval_dataset

    config = finalize_config(config)
    train_cfg = config["train"]
    batch_size = int(train_cfg["local_batch_size"])
    eval_common = _loader_common(config, role="eval")
    components = _build_real_data_components(config, rank, world_size)
    test_delta = components["test_delta"]
    if split == "validation":
        videos = components["val_videos"]
        decoder = components["decoders"]["val"]
    elif split == "test":
        if components["test_videos"] is None:
            raise ValueError(
                "This run has no independent test set: data.test_index was "
                "aliased as the validation split. Evaluate "
                "--split validation instead, or retrain with a dedicated "
                "data.val_index."
            )
        videos = components["test_videos"]
        decoder = components["decoders"]["test"]
    else:
        raise ValueError(f"Unsupported evaluation split: {split!r}")
    dataset = build_eval_dataset(
        videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=max_pairs_per_video,
        decoder=decoder,
    )
    loader = DataLoader(dataset, batch_size=batch_size, **eval_common)
    return loader, components


def _make_dataloaders(
    config: dict, rank: int, world_size: int
) -> LoaderBundle:
    from torch.utils.data import DataLoader, Subset

    from game_cls.data.video_sampler import DeterministicIndexBatchSampler

    data_cfg = config["data"]
    image_spec = ImageSpec.from_config(data_cfg)
    train_cfg = config["train"]
    evaluation_cfg = config["evaluation"]
    batch_size = int(train_cfg["local_batch_size"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    train_common = _loader_common(config, role="train")
    eval_common = _loader_common(config, role="eval")
    probe_pairs_per_video = int(
        evaluation_cfg.get("train_probe_pairs_per_video", 32)
    )
    val_quick_pairs_per_video = int(
        evaluation_cfg.get(
            "val_quick_pairs_per_video",
            evaluation_cfg.get("quick_test_pairs_per_video", 128),
        )
    )

    if data_cfg.get("synthetic", False):
        train_length = max(batch_size * steps_per_epoch * world_size, 128)
        train_dataset = SyntheticPairDataset(
            train_length,
            image_spec,
            config["experiment"]["seed"],
        )
        val_dataset = SyntheticPairDataset(
            64,
            image_spec,
            config["experiment"]["seed"] + 99,
        )
        test_dataset = SyntheticPairDataset(
            64,
            image_spec,
            config["experiment"]["seed"] + 197,
        )
        sampler = DeterministicIndexBatchSampler(
            train_length,
            batch_size,
            steps_per_epoch,
            rank=rank,
            world_size=world_size,
            seed=config["experiment"]["seed"],
        )
        val_full_indices = list(range(rank, len(val_dataset), world_size))
        quick_global = min(
            len(val_dataset),
            val_quick_pairs_per_video * 2,
        )
        val_quick_indices = list(range(rank, quick_global, world_size))
        probe_global = min(
            len(train_dataset),
            probe_pairs_per_video * 2,
        )
        train_probe_indices = list(range(rank, probe_global, world_size))
        test_full_indices = list(range(rank, len(test_dataset), world_size))
        return LoaderBundle(
            train=DataLoader(
                train_dataset, batch_sampler=sampler, **train_common
            ),
            train_probe=DataLoader(
                Subset(train_dataset, train_probe_indices),
                batch_size=batch_size,
                **eval_common,
            ),
            val_quick=DataLoader(
                Subset(val_dataset, val_quick_indices),
                batch_size=batch_size,
                **eval_common,
            ),
            val_full=DataLoader(
                Subset(val_dataset, val_full_indices),
                batch_size=batch_size,
                **eval_common,
            ),
            test_full=DataLoader(
                Subset(test_dataset, test_full_indices),
                batch_size=batch_size,
                **eval_common,
            ),
            sampler=sampler,
            data_summary={
                "storage": "synthetic",
                "train_samples": train_length,
                "train_probe_samples_global": probe_global,
                "val_quick_samples_global": quick_global,
                "val_full_samples_global": len(val_dataset),
                "test_full_samples_global": len(test_dataset),
                "independent_test": True,
            },
        )

    from game_cls.data.augment import ConsistentPairAugment
    from game_cls.data.lazy_pair_dataset import (
        LazyTrainingPairDataset,
        build_eval_dataset,
    )
    from game_cls.data.video_index import video_index_memory_bytes
    from game_cls.data.video_sampler import VideoBalancedPairBatchSampler

    components = _build_real_data_components(config, rank, world_size)
    train_videos = components["train_videos"]
    val_videos = components["val_videos"]
    test_videos = components["test_videos"]
    decoders = components["decoders"]
    test_delta = components["test_delta"]
    delta_probability = components["delta_probability"]
    backend_name = components["backend_name"]
    transform = None
    if config.get("augmentation", {}).get("enabled", True):
        transform = ConsistentPairAugment(config["augmentation"])
    train_dataset = LazyTrainingPairDataset(
        train_videos, transform=transform, decoder=decoders["train"]
    )
    sampler_cfg = config["sampler"]
    dedup_cfg = (config.get("data") or {}).get("deduplication") or {}
    dedup_level = str(dedup_cfg.get("level") or "") or None
    if dedup_level is None:
        legacy = sampler_cfg.get("deduplicate_within_global_batch", True)
        dedup_level = "pair" if legacy else "none"
    sampler = VideoBalancedPairBatchSampler(
        train_videos,
        batch_size,
        steps_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=config["experiment"]["seed"],
        game_alpha=sampler_cfg.get("game_alpha", 0.25),
        class_probability={
            int(key): float(value)
            for key, value in sampler_cfg["class_probability"].items()
        },
        delta_probability=delta_probability,
        dedup_level=dedup_level,
        on_exhaustion=str(dedup_cfg.get("on_exhaustion", "warn_and_relax")),
    )
    # The train probe is a fixed, reproducible, augmentation-free subset of
    # the train split evaluated with the exact validation evaluator.
    train_probe_dataset = build_eval_dataset(
        train_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=probe_pairs_per_video,
        decoder=decoders["train"],
    )
    val_quick_dataset = build_eval_dataset(
        val_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=val_quick_pairs_per_video,
        decoder=decoders["val"],
    )
    val_full_dataset = build_eval_dataset(
        val_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        decoder=decoders["val"],
    )
    test_full_dataset = (
        build_eval_dataset(
            test_videos,
            test_delta,
            rank=rank,
            world_size=world_size,
            decoder=decoders["test"],
        )
        if test_videos is not None
        else None
    )
    global_probe = _distributed_sum_int(len(train_probe_dataset))
    global_quick = _distributed_sum_int(len(val_quick_dataset))
    global_full = _distributed_sum_int(len(val_full_dataset))
    global_test = (
        _distributed_sum_int(len(test_full_dataset))
        if test_full_dataset is not None
        else 0
    )
    if global_quick <= 0 or global_full <= 0:
        raise RuntimeError(
            "Validation index does not contain legal "
            f"delta={test_delta} pairs"
        )
    return LoaderBundle(
        train=DataLoader(
            train_dataset, batch_sampler=sampler, **train_common
        ),
        train_probe=DataLoader(
            train_probe_dataset, batch_size=batch_size, **eval_common
        ),
        val_quick=DataLoader(
            val_quick_dataset, batch_size=batch_size, **eval_common
        ),
        val_full=DataLoader(
            val_full_dataset, batch_size=batch_size, **eval_common
        ),
        test_full=(
            DataLoader(
                test_full_dataset, batch_size=batch_size, **eval_common
            )
            if test_full_dataset is not None
            else None
        ),
        sampler=sampler,
        data_summary={
            "storage": "video_index_lazy_pairs",
            "image_backend": backend_name,
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "test_videos": (
                len(test_videos) if test_videos is not None else None
            ),
            "independent_test": test_videos is not None,
            "video_index_payload_bytes_per_rank_estimate": (
                video_index_memory_bytes(train_videos)
                + video_index_memory_bytes(val_videos)
                + (
                    video_index_memory_bytes(test_videos)
                    if test_videos is not None
                    else 0
                )
            ),
            "train_probe_samples_global": global_probe,
            "val_quick_samples_global": global_quick,
            "val_full_samples_global": global_full,
            "test_full_samples_global": global_test,
            "val_full_index_bytes_this_rank": val_full_dataset.index_nbytes,
        },
    )


def _distributed_sum_int(value: int) -> int:
    if not is_distributed():
        return value
    import torch.distributed as dist

    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return sum(int(item) for item in values)


def _build_scheduler(optimizer, config: dict, total_steps: int):
    import torch

    warmup = int(config.get("warmup_steps", 0))
    min_lr = float(config.get("min_learning_rate", 0.0))
    base_lr = max(group["lr"] for group in optimizer.param_groups)
    min_ratio = min_lr / base_lr if base_lr else 0.0

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _broadcast_object(value, rank: int):
    if not is_distributed():
        return value
    import torch.distributed as dist

    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _gather_random_states(rank: int, world_size: int) -> list[dict] | None:
    local = capture_random_state()
    if not is_distributed():
        return [local]
    import torch.distributed as dist

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


def _normalized_position(
    epoch: int, step_in_epoch: int, steps_per_epoch: int
) -> tuple[int, int]:
    if step_in_epoch >= steps_per_epoch:
        return epoch + 1, 0
    return epoch, step_in_epoch


def _save_all_ranks(
    *,
    output_dir: Path,
    tag: str,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_metrics: dict,
    config: dict,
    sampler,
    evaluation_state: dict,
    rank: int,
    world_size: int,
    force_full_model: bool = False,
) -> None:
    states = _gather_random_states(rank, world_size)
    save_epoch, save_step = _normalized_position(
        epoch, step_in_epoch, int(config["train"]["steps_per_epoch"])
    )
    if rank == 0:
        sampler_state = sampler.state_dict(save_step)
        sampler_state["epoch"] = save_epoch
        checkpoint_cfg = config["checkpoint"]
        state_mode = checkpoint_cfg.get("periodic_state_mode", "full")
        full_model_every = int(
            checkpoint_cfg.get("full_model_every_steps", 0)
        )
        write_model_only = (
            force_full_model
            or state_mode == "full"
            or (full_model_every > 0 and global_step % full_model_every == 0)
        )
        save_checkpoint_pair(
            output_dir / "checkpoints",
            tag,
            model,
            optimizer,
            scheduler,
            scaler,
            save_epoch,
            global_step,
            best_metrics,
            config,
            step_in_epoch=save_step,
            sampler_state=sampler_state,
            rank_random_states=states,
            evaluation_state=evaluation_state,
            state_mode=state_mode,
            write_model_only=write_model_only,
        )


def _selection_score(metrics: dict, evaluation_config: dict) -> tuple[float, bool]:
    metric_name = evaluation_config.get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    minimum_worst = evaluation_config.get("minimum_worst_game_f1")
    eligible = (
        minimum_worst is None
        or float(_metric_value(metrics, "worst_game_f1_at_decision_threshold") or 0.0)
        >= float(minimum_worst)
    )
    if metric_name == "composite":
        weights = evaluation_config.get("selection_weights", {})
        components = {
            "global_f1": float(
                _metric_value(metrics, "global_f1_at_decision_threshold") or 0.0
            ),
            "macro_game_f1": float(
                _metric_value(metrics, "macro_game_f1_at_decision_threshold")
                or 0.0
            ),
            "worst_game_f1": float(
                _metric_value(metrics, "worst_game_f1_at_decision_threshold")
                or 0.0
            ),
        }
        total_weight = sum(float(weights.get(key, 0.0)) for key in components)
        if total_weight <= 0:
            raise ValueError(
                "evaluation.selection_weights must contain a positive weight"
            )
        score = sum(
            components[key] * float(weights.get(key, 0.0))
            for key in components
        ) / total_weight
    else:
        value = _metric_value(metrics, metric_name)
        if value is None:
            raise KeyError(
                f"Selection metric is absent from evaluation output: {metric_name}"
            )
        score = float(value)
    return score, eligible


def _annotate_selection(metrics: dict, evaluation_config: dict) -> dict:
    score, eligible = _selection_score(metrics, evaluation_config)
    metrics["selection_metric"] = evaluation_config.get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    metrics["selection_score"] = score
    metrics["selection_eligible"] = eligible
    metrics["minimum_worst_game_f1"] = evaluation_config.get(
        "minimum_worst_game_f1"
    )
    return metrics


def _is_better_model(
    candidate: dict, incumbent: dict, evaluation_config: dict
) -> bool:
    candidate_score, candidate_eligible = _selection_score(
        candidate, evaluation_config
    )
    if not candidate_eligible:
        return False
    if not incumbent:
        return True
    incumbent_score, incumbent_eligible = _selection_score(
        incumbent, evaluation_config
    )
    return not incumbent_eligible or candidate_score > incumbent_score


def _save_best_enabled(checkpoint_config: dict) -> bool:
    return bool(
        checkpoint_config.get(
            "save_best_selection",
            checkpoint_config.get("save_best_test_f1", True),
        )
    )


def _run_evaluation(
    *,
    kind: str,
    model,
    dataloader,
    device,
    config: dict,
    output_dir: Path,
    global_step: int,
    rank: int,
    world_size: int,
    total_steps: int | None = None,
    evaluation_state: dict | None = None,
) -> EvaluationOutput:
    if kind not in _EVALUATION_ROLES:
        raise ValueError(f"Unknown evaluation kind: {kind}")
    role = _EVALUATION_ROLES[kind]
    scope = _EVALUATION_SCOPES[kind]
    report_dir = output_dir / "reports" / f"{kind}_step_{global_step:08d}"
    prepare_evaluation_directory(report_dir, rank)
    distributed_barrier()
    loss_cfg = config["loss"]
    threshold_weight = _threshold_weight_for_eval(
        config,
        global_step,
        max(1, total_steps if total_steps is not None else global_step),
    )
    result = evaluate(
        unwrap_model(model),
        dataloader,
        device,
        config["evaluation"].get("threshold", 0.99),
        checkpoint_step=global_step,
        distributed=is_distributed(),
        rank=rank,
        world_size=world_size,
        evaluation_kind=("quick" if scope == "quick" else kind),
        report_dir=report_dir,
        full_auc_mode=config["evaluation"].get("full_auc_mode", "histogram"),
        auc_histogram_bins=int(
            config["evaluation"].get("auc_histogram_bins", 4096)
        ),
        quick_error_limit=int(
            config["evaluation"].get("quick_save_error_limit", 200)
        ),
        amp=bool(
            config["evaluation"].get(
                "amp", config["device"].get("amp", False)
            )
        ),
        amp_dtype=str(
            config["evaluation"].get(
                "amp_dtype",
                config["device"].get("amp_dtype", "bfloat16"),
            )
        ),
        parquet_row_group_size=int(
            config["evaluation"].get(
                "parquet_row_group_size", 4096
            )
        ),
        group_catalogs=getattr(
            getattr(dataloader, "dataset", None),
            "group_catalogs",
            None,
        ),
        threshold_loss_weight=threshold_weight,
        cross_entropy_weight=float(
            loss_cfg.get("cross_entropy_weight", 1.0)
        ),
        threshold_safety_margin=float(
            loss_cfg.get("threshold_safety_margin", 0.20)
        ),
        threshold_temperature=float(
            loss_cfg.get("threshold_temperature", 0.50)
        ),
    )
    distributed_barrier()
    if rank == 0:
        metrics = dict(result.metrics or {})
        metrics.update(
            {
                "evaluation_kind": kind,
                "evaluation_role": role,
                "evaluation_scope": scope,
                "checkpoint_step": global_step,
                "evaluation_amp": bool(
                    config["evaluation"].get(
                        "amp", config["device"].get("amp", False)
                    )
                ),
                "evaluation_amp_dtype": str(
                    config["evaluation"].get(
                        "amp_dtype",
                        config["device"].get("amp_dtype", "bfloat16"),
                    )
                ),
            }
        )
        _annotate_selection(metrics, config["evaluation"])
        write_evaluation_report(
            report_dir,
            metrics,
            result.grouped_metrics,
            merge_shards=True,
            lightweight=scope in ("quick", "probe"),
            html_max_errors=int(
                config["evaluation"].get("html_max_errors_per_group", 200)
            ),
            preview_decoder=getattr(
                getattr(dataloader, "dataset", None), "decoder", None
            ),
        )
        history_extra: dict = {}
        if kind == "val_full" and evaluation_state is not None:
            probe_metrics = evaluation_state.get(
                "last_train_probe_metrics"
            ) or {}
            if probe_metrics.get("checkpoint_step") == global_step:
                probe_ce = probe_metrics.get("cross_entropy")
                val_ce = metrics.get("cross_entropy")
                probe_score = probe_metrics.get("selection_score")
                val_score = metrics.get("selection_score")
                if isinstance(probe_ce, (int, float)) and isinstance(
                    val_ce, (int, float)
                ):
                    history_extra["generalization_ce_gap"] = float(
                        val_ce
                    ) - float(probe_ce)
                if isinstance(probe_score, (int, float)) and isinstance(
                    val_score, (int, float)
                ):
                    history_extra["generalization_score_gap"] = float(
                        probe_score
                    ) - float(val_score)
        _append_evaluation_history(
            output_dir,
            _evaluation_history_record(
                metrics,
                kind=kind,
                global_step=global_step,
                extra=history_extra or None,
            ),
        )
        result.metrics = metrics
    distributed_barrier()
    return result


def run_training(
    config: dict[str, Any],
    run_meta: dict[str, Any] | None = None,
    on_run_dir: Callable[[Path], None] | None = None,
) -> dict:
    """Train the dual-frame classifier.

    ``experiment.run_mode`` controls output placement:

    * ``fixed`` (default): write directly into ``experiment.output_dir``
      (legacy behavior, exact resume and tests rely on it).
    * ``unique``: treat ``experiment.output_dir`` as a runs ROOT and
      allocate a fresh, never-overwritten timestamped run directory under
      it. Rank 0 allocates the directory and broadcasts it, so distributed
      launches agree on a single run.

    ``run_meta`` carries launch facts (command, environment, checkpoint
    hash) for the manifest; ``on_run_dir`` is a rank-0 callback fired as
    soon as the run directory exists (used to attach console capture).
    """
    import torch

    config = finalize_config(config)
    validate_training_config(config)
    image_spec = ImageSpec.from_config(config["data"])
    rank, world_size, local_rank, device = initialize_runtime(config)
    run_mode = str(config["experiment"].get("run_mode", "fixed"))
    meta = run_meta or {}
    resuming = bool(
        config["train"].get("resume_path") or meta.get("resumed_from")
    )
    if resuming and meta.get("runs_root"):
        # Resume writes index/status updates into the ORIGINAL runs root,
        # never into the run directory itself.
        runs_root = Path(meta["runs_root"])
    else:
        runs_root = Path(config["experiment"]["output_dir"])
    started_wall = time.time()
    run_id: str | None = None
    output_dir: Path | None = None
    global_step = 0
    try:
        seed = int(config["experiment"]["seed"])
        _seed_everything(seed + rank)
        config_output = Path(config["experiment"]["output_dir"])
        if run_mode == "unique":
            allocation: tuple[str, str] | None = None
            if rank == 0:
                allocated, allocated_id = allocate_run_dir(
                    runs_root, config["experiment"].get("name", "run")
                )
                allocation = (str(allocated), allocated_id)
            allocation = _broadcast_object(allocation, rank)
            output_dir = Path(allocation[0])
            run_id = allocation[1]
        else:
            # fixed/resume: write into the configured directory itself; the
            # runs root (possibly different on resume) only hosts the index.
            output_dir = config_output
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            _write_resolved_config(output_dir, config)
            effective_run_id = _write_run_manifest(
                output_dir,
                config=config,
                run_meta=run_meta,
                run_id=run_id,
                run_mode=run_mode,
                world_size=world_size,
            )
            if effective_run_id is not None:
                run_id = effective_run_id
            status_fields: dict[str, Any] = {
                "state": STATE_RUNNING,
                "run_id": run_id,
            }
            existing_status = _read_status(output_dir)
            if not existing_status.get("started"):
                status_fields["started"] = _iso_now()
            if config["train"].get("resume_path") or (run_meta or {}).get(
                "resumed_from"
            ):
                status_fields["resumed_at"] = _iso_now()
            update_status(output_dir, **status_fields)
            append_run_index(
                runs_root,
                {
                    "run_id": run_id,
                    "name": config["experiment"].get("name"),
                    "output_dir": str(output_dir),
                    "state": STATE_RUNNING,
                    "started": status_fields.get("started"),
                    "global_step": global_step,
                },
            )
            resume_path = config["train"].get("resume_path")
            if resume_path:
                meta = run_meta or {}
                append_resume_event(
                    output_dir,
                    checkpoint=resume_path,
                    command=meta.get("command"),
                    resume_type=str(meta.get("resume_type", "exact")),
                    config_diffs=list(meta.get("resume_config_diffs", [])),
                )
            if on_run_dir is not None:
                on_run_dir(output_dir)

        model = build_model(config["model"])
        checkpoint_path = config["model"].get("checkpoint_path")
        if checkpoint_path:
            report = load_model_checkpoint(model, checkpoint_path)
            if not config["data"].get("synthetic", False) and config["model"].get(
                "require_pretrained_backbone", True
            ):
                coverage = validate_production_load(
                    model,
                    report,
                    trainable_name_contains=config["model"].get(
                        "trainable_name_contains", "cls"
                    ),
                )
            else:
                coverage = None
            if rank == 0:
                print(f"Loaded {len(report.loaded)} model tensors")
                if coverage is not None:
                    print(f"Frozen backbone checkpoint coverage: {coverage:.2%}")
                print(f"Missing: {report.missing}")
                print(f"Unexpected: {report.unexpected}")
                print(f"Shape mismatch: {report.shape_mismatch}")
        summary = configure_trainable_parameters(
            model, config["model"].get("trainable_name_contains", "cls")
        )
        _set_train_mode(model, config["model"])
        model.to(device)
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel

            # CPU DDP must NOT pass device_ids (it has no CUDA devices);
            # only CUDA/NPU rank-to-device binding is valid.
            ddp_kwargs: dict[str, Any] = {}
            if device.type in ("cuda", "npu"):
                ddp_kwargs["device_ids"] = [local_rank]
            model = DistributedDataParallel(
                model,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                **ddp_kwargs,
            )
        frozen_snapshot = None
        if rank == 0:
            print("Trainable parameters:")
            for name in summary.trainable_names:
                print(f"  {name}")
            print(
                f"Trainable={summary.trainable_count:,} Frozen={summary.frozen_count:,} "
                f"Ratio={summary.trainable_ratio:.4%}"
            )

        loaders = _make_dataloaders(config, rank, world_size)
        if rank == 0:
            for warning in split_role_warnings(config):
                print(f"[WARNING] {warning}", flush=True)
            print("Data pipeline:", json.dumps(loaders.data_summary, ensure_ascii=False))
            train_workers = int(
                _dataloader_option(
                    config, "train", "num_workers", 0
                )
            )
            eval_workers = int(
                _dataloader_option(config, "eval", "num_workers", 0)
            )
            context = _dataloader_option(
                config,
                "train",
                "multiprocessing_context",
                None,
            )
            if (
                context is None
                and train_workers > 0
                and str(config["device"]["accelerator"]) == "npu"
            ):
                context = "spawn"
            timeout = _dataloader_option(
                config, "train", "timeout_seconds", 180
            )
            print(
                "[DATALOADER] "
                f"train_workers={train_workers} "
                f"eval_workers={eval_workers} "
                f"context={context} "
                f"timeout={timeout}",
                flush=True,
            )
        train_cfg = config["train"]
        total_steps = int(
            train_cfg.get("max_steps")
            or int(train_cfg["epochs"]) * int(train_cfg["steps_per_epoch"])
        )
        optimizer = torch.optim.AdamW(
            build_optimizer_parameter_groups(
                model, float(config["optimizer"]["weight_decay"])
            ),
            lr=config["optimizer"]["learning_rate"],
        )
        scheduler = _build_scheduler(optimizer, config["scheduler"], total_steps)
        use_amp = bool(config["device"].get("amp", False))
        scaler = torch.amp.GradScaler(
            device.type, enabled=use_amp and config["device"]["amp_dtype"] == "float16"
        )
        global_step = 0
        epoch = 0
        step_in_epoch = 0
        best_metrics: dict = {}
        best_val_loss_metrics: dict = {}
        best_worst_game_metrics: dict = {}
        evaluation_state = {
            "quick_test_count": 0,
            "full_test_count": 0,
            "val_quick_count": 0,
            "val_full_count": 0,
            "train_probe_count": 0,
            "last_full_metrics": {},
            "last_train_probe_metrics": {},
            "best_observed_dev_test_metrics": {},
            "best_validation_metrics": {},
            "best_val_loss_metrics": {},
            "best_worst_game_metrics": {},
            "early_stopping": _early_stopping_defaults(),
            "train_delta_history": [],
            "topk_registry": [],
        }
        resume_path = train_cfg.get("resume_path")
        if resume_path:
            checkpoint = restore_training_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                scaler,
                expected_base_checkpoint=config["model"].get("checkpoint_path"),
            )
            global_step = int(checkpoint.get("global_step", 0))
            sampler_state = checkpoint.get("sampler_state", {})
            epoch = int(
                sampler_state.get(
                    "epoch", checkpoint.get("sampler_epoch", checkpoint.get("epoch", 0))
                )
            )
            step_in_epoch = int(
                sampler_state.get(
                    "step_in_epoch", checkpoint.get("step_in_epoch", 0)
                )
            )
            best_metrics = dict(checkpoint.get("best_metrics", {}))
            evaluation_state.update(checkpoint.get("evaluation_state", {}))
            # Old checkpoints predate the early-stopping state; resume must
            # not recompute patience from scratch for the restored run.
            evaluation_state.setdefault(
                "early_stopping", _early_stopping_defaults()
            )
            evaluation_state.setdefault(
                "last_train_probe_metrics", {}
            )
            evaluation_state.setdefault("best_validation_metrics", {})
            evaluation_state.setdefault("best_val_loss_metrics", {})
            evaluation_state.setdefault("best_worst_game_metrics", {})
            evaluation_state.setdefault("topk_registry", [])
            rank_states = checkpoint.get("rank_random_states")
            if rank_states and rank < len(rank_states):
                restore_random_state(rank_states[rank])
            else:
                restore_random_state(checkpoint.get("random_state", {}))
            if rank == 0:
                print(
                    f"Resumed from {resume_path} at step={global_step}, "
                    f"epoch={epoch}, step_in_epoch={step_in_epoch}"
                )
            best_val_loss_metrics = dict(
                evaluation_state.get("best_val_loss_metrics", {})
            )
            best_worst_game_metrics = dict(
                evaluation_state.get("best_worst_game_metrics", {})
            )
        if global_step >= total_steps:
            raise ValueError(
                f"Resume step {global_step} is not below target max step {total_steps}"
            )
        if config["train"].get("verify_frozen_parameters", False):
            frozen_snapshot = snapshot_frozen_parameters(model)
        stop_after_steps = train_cfg.get("stop_after_steps")
        run_until_step = (
            min(total_steps, int(stop_after_steps))
            if stop_after_steps is not None
            else total_steps
        )
        if global_step >= run_until_step:
            raise ValueError(
                f"Current step {global_step} is not below this run's stop step "
                f"{run_until_step}"
            )

        started = time.perf_counter()
        last_batch_finished = started
        processed_samples = 0
        log_interval_start = started
        log_interval_samples = 0
        evaluation_seconds = 0.0
        checkpoint_seconds = 0.0
        metrics_path = output_dir / "train_metrics.jsonl"
        if rank == 0 and not resume_path:
            metrics_path.write_text("", encoding="utf-8")
            history_path = output_dir / "metrics" / "evaluation.jsonl"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            if not history_path.exists():
                history_path.write_text("", encoding="utf-8")
        started_iso = _iso_now()
        decision_cutoff = probability_threshold_to_margin(
            float(config["decision"]["threshold"])
        )
        interval_accum = _new_interval_accumulator(device)
        tb_writer = None
        if rank == 0 and bool(
            config["evaluation"].get("tensorboard_live", False)
        ):
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(
                    log_dir=str(output_dir / "tensorboard")
                )
            except ImportError:
                tb_writer = None
        early_stop_requested = False
        timing = {
            "host_data_wait": 0.0,
            "host_h2d_enqueue": 0.0,
            "host_forward_enqueue": 0.0,
            "host_backward_enqueue": 0.0,
            "host_optimizer_enqueue": 0.0,
        }
        timing_steps = 0
        input_shape_validated = False
        first_batch_wait_started = time.perf_counter()
        first_batch_logged = False
        if rank == 0:
            print(
                "[DATALOADER] starting train workers and waiting for first batch",
                flush=True,
            )
        while global_step < run_until_step and not early_stop_requested:
            loaders.sampler.set_epoch(epoch, start_step=step_in_epoch)
            _set_train_mode(model, config["model"])
            yielded = False
            for batch in loaders.train:
                yielded = True
                batch_ready = time.perf_counter()
                if not first_batch_logged:
                    if rank == 0:
                        print(
                            "[DATALOADER] first batch ready: "
                            f"wait={batch_ready - first_batch_wait_started:.3f}s "
                            f"shape={tuple(batch['images'].shape)} "
                            f"dtype={batch['images'].dtype}",
                            flush=True,
                        )
                    first_batch_logged = True
                timing["host_data_wait"] += batch_ready - last_batch_finished
                transfer_started = time.perf_counter()
                images = batch["images"]
                if not input_shape_validated:
                    image_spec.validate_pair_batch_shape(images.shape)
                    input_shape_validated = True
                images = images.to(device, non_blocking=True)
                if images.dtype == torch.uint8:
                    compute_dtype = (
                        torch.bfloat16
                        if use_amp and config["device"]["amp_dtype"] == "bfloat16"
                        else torch.float16
                        if use_amp and config["device"]["amp_dtype"] == "float16"
                        else torch.float32
                    )
                    images = images.to(compute_dtype).div_(255.0)
                labels = batch["labels"].to(device, non_blocking=True)
                timing["host_h2d_enqueue"] += (
                    time.perf_counter() - transfer_started
                )
                optimizer.zero_grad(set_to_none=True)
                forward_started = time.perf_counter()
                with autocast_context(
                    device, use_amp, config["device"].get("amp_dtype", "float16")
                ):
                    logits = model(images[:, 0], images[:, 1])
                    if logits.ndim != 2 or logits.shape[1] != 2:
                        raise ValueError(
                            f"Model must return [B,2], got {tuple(logits.shape)}"
                        )
                    loss, components = combined_loss(
                        logits, labels, config["loss"], global_step, total_steps
                    )
                timing["host_forward_enqueue"] += (
                    time.perf_counter() - forward_started
                )
                backward_started = time.perf_counter()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.get("gradient_clip_norm", 5.0),
                )
                timing["host_backward_enqueue"] += (
                    time.perf_counter() - backward_started
                )
                optimizer_started = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                timing["host_optimizer_enqueue"] += (
                    time.perf_counter() - optimizer_started
                )
                step_samples = (
                    int(train_cfg["local_batch_size"]) * world_size
                )
                processed_samples += step_samples
                log_interval_samples += step_samples
                global_step += 1
                step_in_epoch += 1
                timing_steps += 1

                # Sample-weighted interval accumulators (exact statistics,
                # distributed-reduced at log time; no per-step sync).
                batch_samples = int(labels.numel())
                with torch.no_grad():
                    eval_margin = (
                        logits.detach().float()[:, 1]
                        - logits.detach().float()[:, 0]
                    )
                    predictions = eval_margin > decision_cutoff
                    interval_accum["counts"].add_(
                        torch.stack(
                            (
                                (predictions & (labels == 1)).sum(),
                                (predictions & (labels == 0)).sum(),
                                ((~predictions) & (labels == 1)).sum(),
                                ((~predictions) & (labels == 0)).sum(),
                            )
                        ).to(torch.int64)
                    )
                interval_accum["loss_sum"].add_(
                    loss.detach().double().mul_(batch_samples)
                )
                interval_accum["ce_sum"].add_(
                    components["cross_entropy"].double().mul_(batch_samples)
                )
                interval_accum["threshold_sum"].add_(
                    components["threshold_loss"].double().mul_(
                        batch_samples
                    )
                )
                interval_accum["threshold_weight_sum"] += (
                    float(components["threshold_weight"]) * batch_samples
                )
                interval_accum["samples"] += batch_samples

                log_every = int(train_cfg["log_every_steps"])

                evaluation_cfg = config["evaluation"]
                probe_every = int(
                    evaluation_cfg.get("train_probe_every_steps", 0)
                )
                quick_every = int(
                    evaluation_cfg.get("val_quick_every_steps", 0)
                )
                full_every = int(
                    evaluation_cfg.get("val_full_every_steps", 0)
                )
                run_full = bool(
                    full_every and global_step % full_every == 0
                )
                run_probe = bool(
                    probe_every
                    and global_step % probe_every == 0
                    and loaders.train_probe is not None
                )
                run_quick = bool(
                    quick_every
                    and global_step % quick_every == 0
                    and not run_full
                )
                if run_probe:
                    evaluation_started = time.perf_counter()
                    result = _run_evaluation(
                        kind="train_probe",
                        model=model,
                        dataloader=loaders.train_probe,
                        device=device,
                        config=config,
                        output_dir=output_dir,
                        global_step=global_step,
                        rank=rank,
                        world_size=world_size,
                        total_steps=total_steps,
                    )
                    evaluation_state["train_probe_count"] += 1
                    if rank == 0:
                        evaluation_state["last_train_probe_metrics"] = (
                            result.metrics or {}
                        )
                        _tb_write_scalars(
                            tb_writer,
                            "eval_train_probe",
                            result.metrics or {},
                            global_step,
                        )
                    _set_train_mode(model, config["model"])
                    evaluation_seconds += (
                        time.perf_counter() - evaluation_started
                    )
                if run_quick:
                    evaluation_started = time.perf_counter()
                    result = _run_evaluation(
                        kind="val_quick",
                        model=model,
                        dataloader=loaders.val_quick,
                        device=device,
                        config=config,
                        output_dir=output_dir,
                        global_step=global_step,
                        rank=rank,
                        world_size=world_size,
                        total_steps=total_steps,
                    )
                    evaluation_state["quick_test_count"] += 1
                    evaluation_state["val_quick_count"] += 1
                    if rank == 0:
                        _tb_write_scalars(
                            tb_writer,
                            "eval_val_quick",
                            result.metrics or {},
                            global_step,
                        )
                    early_cfg = config.get("early_stopping") or {}
                    if early_cfg.get(
                        "enabled", False
                    ) and not early_cfg.get("full_validation_only", True):
                        quick_metrics = _broadcast_object(
                            result.metrics, rank
                        )
                        early_state = evaluation_state["early_stopping"]
                        if _update_early_stopping(
                            early_state,
                            early_cfg,
                            quick_metrics or {},
                            global_step,
                        ):
                            early_state["stop_reason"] = (
                                "validation_plateau_quick"
                            )
                            early_state["stopped_at_step"] = global_step
                            early_stop_requested = True
                            if rank == 0:
                                print(
                                    "[EARLY-STOP] quick validation plateau: "
                                    f"monitor={early_cfg.get('monitor', 'selection_score')} "
                                    f"bad_evaluations={early_state['bad_evaluation_count']} "
                                    f"best_step={early_state['best_step']} "
                                    f"step={global_step}",
                                    flush=True,
                                )
                                update_status(
                                    output_dir,
                                    early_stopped=True,
                                    early_stopped_step=global_step,
                                )
                    _set_train_mode(model, config["model"])
                    evaluation_seconds += (
                        time.perf_counter() - evaluation_started
                    )

                is_best = False
                loss_improved = False
                worst_improved = False
                if run_full:
                    evaluation_started = time.perf_counter()
                    result = _run_evaluation(
                        kind="val_full",
                        model=model,
                        dataloader=loaders.val_full,
                        device=device,
                        config=config,
                        output_dir=output_dir,
                        global_step=global_step,
                        rank=rank,
                        world_size=world_size,
                        total_steps=total_steps,
                        evaluation_state=evaluation_state,
                    )
                    evaluation_state["full_test_count"] += 1
                    evaluation_state["val_full_count"] += 1
                    metrics = _broadcast_object(result.metrics, rank)
                    evaluation_state["last_full_metrics"] = metrics
                    if rank == 0:
                        _tb_write_scalars(
                            tb_writer,
                            "eval_validation",
                            metrics or {},
                            global_step,
                        )
                    is_best = _is_better_model(
                        metrics, best_metrics, config["evaluation"]
                    )
                    if is_best:
                        best_metrics = metrics
                        evaluation_state["best_observed_dev_test_metrics"] = metrics
                        evaluation_state["best_validation_metrics"] = metrics
                    val_ce = (
                        metrics.get("cross_entropy")
                        if isinstance(metrics, dict)
                        else None
                    )
                    loss_improved = isinstance(val_ce, (int, float)) and (
                        not best_val_loss_metrics
                        or float(val_ce)
                        < float(
                            best_val_loss_metrics.get("cross_entropy")
                        )
                    )
                    if loss_improved:
                        best_val_loss_metrics = metrics
                        evaluation_state["best_val_loss_metrics"] = metrics
                    worst_f1 = (
                        _metric_value(metrics, "worst_game_f1_at_decision_threshold")
                        if isinstance(metrics, dict)
                        else None
                    )
                    worst_improved = isinstance(
                        worst_f1, (int, float)
                    ) and (
                        not best_worst_game_metrics
                        or float(worst_f1)
                        > float(
                            _metric_value(
                                best_worst_game_metrics,
                                "worst_game_f1_at_decision_threshold",
                            )
                            or 0.0
                        )
                    )
                    if worst_improved:
                        best_worst_game_metrics = metrics
                        evaluation_state[
                            "best_worst_game_metrics"
                        ] = metrics
                    early_cfg = config.get("early_stopping") or {}
                    if early_cfg.get("enabled", False):
                        early_state = evaluation_state["early_stopping"]
                        if _update_early_stopping(
                            early_state,
                            early_cfg,
                            metrics or {},
                            global_step,
                        ):
                            early_state["stop_reason"] = (
                                "validation_plateau"
                            )
                            early_state["stopped_at_step"] = global_step
                            early_stop_requested = True
                            if rank == 0:
                                print(
                                    "[EARLY-STOP] validation plateau: "
                                    f"monitor={early_cfg.get('monitor', 'selection_score')} "
                                    f"bad_evaluations={early_state['bad_evaluation_count']} "
                                    f"best_step={early_state['best_step']} "
                                    f"step={global_step}",
                                    flush=True,
                                )
                                update_status(
                                    output_dir,
                                    early_stopped=True,
                                    early_stopped_step=global_step,
                                )
                    if rank == 0:
                        try:
                            overview_html = render_overview_html(
                                run_dir=output_dir,
                                run_id=run_id,
                                state=STATE_RUNNING,
                                started=started_iso,
                                finished=None,
                                duration_seconds=(
                                    time.time() - started_wall
                                ),
                                config=config,
                                summary_payload=None,
                            )
                            atomic_write_text(
                                output_dir / "overview.html", overview_html
                            )
                        except OSError:
                            pass
                    _set_train_mode(model, config["model"])
                    evaluation_seconds += (
                        time.perf_counter() - evaluation_started
                    )

                save_every = int(
                    config["checkpoint"].get("save_last_every_steps", 0)
                )
                save_last_due = bool(
                    save_every and global_step % save_every == 0
                )
                any_improved = is_best or loss_improved or worst_improved
                checkpoint_started = None
                topk_requested = bool(
                    run_full
                    and int(config["checkpoint"].get("save_topk", 0)) > 0
                )
                if any_improved or topk_requested:
                    checkpoint_started = time.perf_counter()
                    _save_all_ranks(
                        output_dir=output_dir,
                        tag="last",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        best_metrics=best_metrics,
                        config=config,
                        sampler=loaders.sampler,
                        evaluation_state=evaluation_state,
                        rank=rank,
                        world_size=world_size,
                        force_full_model=True,
                    )
                    if rank == 0:
                        checkpoint_cfg = config["checkpoint"]
                        if is_best and _save_best_enabled(checkpoint_cfg):
                            clone_checkpoint_pair(
                                output_dir / "checkpoints",
                                "last",
                                "best_selection",
                            )
                            clone_checkpoint_pair(
                                output_dir / "checkpoints",
                                "last",
                                "best_observed_dev_test_selection",
                            )
                        if loss_improved and checkpoint_cfg.get(
                            "save_best_val_loss", True
                        ):
                            clone_checkpoint_pair(
                                output_dir / "checkpoints",
                                "last",
                                "best_val_loss",
                            )
                        if worst_improved and checkpoint_cfg.get(
                            "save_best_worst_game", True
                        ):
                            clone_checkpoint_pair(
                                output_dir / "checkpoints",
                                "last",
                                "best_worst_game",
                            )
                        if topk_requested:
                            _maybe_save_topk(
                                output_dir=output_dir,
                                checkpoint_cfg=checkpoint_cfg,
                                metrics=metrics or {},
                                global_step=global_step,
                                evaluation_state=evaluation_state,
                            )
                elif save_last_due:
                    checkpoint_started = time.perf_counter()
                    _save_all_ranks(
                        output_dir=output_dir,
                        tag="last",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        best_metrics=best_metrics,
                        config=config,
                        sampler=loaders.sampler,
                        evaluation_state=evaluation_state,
                        rank=rank,
                        world_size=world_size,
                    )
                if checkpoint_started is not None:
                    checkpoint_seconds += (
                        time.perf_counter() - checkpoint_started
                    )

                should_log = (
                    global_step % log_every == 0
                    or global_step >= run_until_step
                )
                if should_log:
                    _synchronize_device_for_metrics(device)
                    interval_metrics = _reduce_interval_accumulator(
                        interval_accum, device
                    )
                    if rank == 0:
                        now = time.perf_counter()
                        interval_seconds = now - log_interval_start
                        interval_steps = max(1, timing_steps)
                        elapsed = now - started
                        averages = {
                            key: value / interval_steps
                            for key, value in timing.items()
                        }
                        loss_value = float(loss.detach().item())
                        ce_value = float(
                            components["cross_entropy"].item()
                        )
                        threshold_loss_value = float(
                            components["threshold_loss"].item()
                        )
                        grad_norm_value = float(
                            grad_norm.detach().item()
                        )
                        learning_rate = max(
                            float(group["lr"])
                            for group in optimizer.param_groups
                        )
                        interval_samples_per_second = (
                            log_interval_samples
                            / max(interval_seconds, 1e-9)
                        )
                        interval_step_time = (
                            interval_seconds / interval_steps
                        )
                        data_wait_seconds = timing["host_data_wait"]
                        data_wait_ratio = (
                            data_wait_seconds
                            / max(interval_seconds, 1e-9)
                        )
                        wall_samples_per_second = (
                            processed_samples / max(elapsed, 1e-9)
                        )
                        metrics_payload = {
                            "step": global_step,
                            "total_steps": total_steps,
                            # Raw last-batch values (kept for backward
                            # compatibility; noisy by nature).
                            "loss": loss_value,
                            "ce": ce_value,
                            "threshold_loss": threshold_loss_value,
                            "threshold_weight": float(
                                components["threshold_weight"]
                            ),
                            # Exact sample-weighted interval averages,
                            # distributed-reduced: the canonical curves.
                            "interval_loss": interval_metrics[
                                "interval_loss"
                            ],
                            "interval_ce": interval_metrics["interval_ce"],
                            "interval_threshold_loss": interval_metrics[
                                "interval_threshold_loss"
                            ],
                            "interval_threshold_weight": interval_metrics[
                                "interval_threshold_weight"
                            ],
                            "interval_accuracy": interval_metrics[
                                "interval_accuracy"
                            ],
                            "interval_positive_recall_tau099": (
                                interval_metrics[
                                    "interval_positive_recall_tau099"
                                ]
                            ),
                            "interval_negative_specificity_tau099": (
                                interval_metrics[
                                    "interval_negative_specificity_tau099"
                                ]
                            ),
                            "interval_samples": interval_metrics[
                                "interval_samples"
                            ],
                            "interval_samples_per_second": (
                                interval_samples_per_second
                            ),
                            "interval_seconds": interval_seconds,
                            "interval_step_time": interval_step_time,
                            "data_wait_seconds": data_wait_seconds,
                            "data_wait_ratio": data_wait_ratio,
                            "learning_rate": learning_rate,
                            "grad_norm": grad_norm_value,
                            "evaluation_seconds": evaluation_seconds,
                            "checkpoint_seconds": checkpoint_seconds,
                            "wall_samples_per_second": (
                                wall_samples_per_second
                            ),
                            "host_enqueue_timing": averages,
                        }
                        _append_training_metrics(
                            metrics_path, metrics_payload
                        )
                        _tb_write_scalars(
                            tb_writer,
                            "train",
                            metrics_payload,
                            global_step,
                        )
                        update_status(
                            output_dir,
                            state=STATE_RUNNING,
                            step=global_step,
                            loss=interval_metrics["interval_loss"],
                            learning_rate=learning_rate,
                        )
                        print(
                            f"step={global_step}/{total_steps} "
                            f"loss={interval_metrics['interval_loss']:.6f} "
                            f"ce={interval_metrics['interval_ce']:.6f} "
                            f"threshold_loss="
                            f"{interval_metrics['interval_threshold_loss']:.6f} "
                            f"threshold_weight="
                            f"{interval_metrics['interval_threshold_weight']:.4f} "
                            f"accuracy="
                            f"{interval_metrics['interval_accuracy']:.4f} "
                            f"interval_samples/s="
                            f"{interval_samples_per_second:.2f} "
                            f"interval_step_time="
                            f"{interval_step_time:.4f}s "
                            f"data_wait_ratio={data_wait_ratio:.2%} "
                            f"learning_rate={learning_rate:.8g} "
                            f"grad_norm={grad_norm_value:.6f} "
                            f"evaluation_seconds="
                            f"{evaluation_seconds:.3f} "
                            f"checkpoint_seconds="
                            f"{checkpoint_seconds:.3f} "
                            f"wall_samples/s="
                            f"{wall_samples_per_second:.2f} "
                            f"host_enqueue_timing={averages}",
                            flush=True,
                        )
                    interval_accum = _new_interval_accumulator(device)
                    log_interval_start = time.perf_counter()
                    log_interval_samples = 0
                    evaluation_seconds = 0.0
                    checkpoint_seconds = 0.0
                    timing = {key: 0.0 for key in timing}
                    timing_steps = 0
                last_batch_finished = time.perf_counter()
                if global_step >= run_until_step or early_stop_requested:
                    break
            if not yielded and step_in_epoch < int(train_cfg["steps_per_epoch"]):
                raise RuntimeError("Training sampler yielded no batches")
            if step_in_epoch >= int(train_cfg["steps_per_epoch"]):
                delta_counts = getattr(
                    loaders.sampler, "last_epoch_delta_counts", None
                )
                if delta_counts:
                    total_sampled = sum(delta_counts.values())
                    distribution = {
                        str(delta): count / total_sampled
                        for delta, count in sorted(delta_counts.items())
                    }
                    evaluation_state["train_delta_history"].append(
                        {
                            "epoch": epoch,
                            "counts": dict(delta_counts),
                            "distribution": distribution,
                            "by_game_label_delta": [
                                {
                                    "game": game,
                                    "label": label,
                                    "delta": delta,
                                    "count": count,
                                }
                                for (
                                    game,
                                    label,
                                    delta,
                                ), count in sorted(
                                    getattr(
                                        loaders.sampler,
                                        "last_epoch_game_label_delta_counts",
                                        {},
                                    ).items()
                                )
                            ],
                        }
                    )
                    dedup_failures = getattr(
                        loaders.sampler, "last_epoch_dedup_failures", 0
                    )
                    if dedup_failures:
                        evaluation_state.setdefault(
                            "dedup_failures_by_epoch", []
                        ).append(
                            {"epoch": epoch, "failures": int(dedup_failures)}
                        )
                    if rank == 0:
                        if dedup_failures:
                            print(
                                "[WARNING] deduplication relaxed "
                                f"{dedup_failures} times this epoch; "
                                "increase the data pool or lower the batch "
                                "size.",
                                flush=True,
                            )
                        print(
                            "Observed train delta distribution:",
                            json.dumps(distribution, ensure_ascii=False),
                        )
                epoch += 1
                step_in_epoch = 0

        final_is_best = False
        restored_best = False
        final_metrics: dict = {}
        early_cfg = config.get("early_stopping") or {}
        best_selection_path = (
            output_dir / "checkpoints" / "model_best_selection.pth"
        )
        if (
            early_cfg.get("restore_best", True)
            and evaluation_state["early_stopping"].get("best_step")
            and best_selection_path.is_file()
        ):
            state_dict = torch.load(
                best_selection_path, map_location="cpu", weights_only=True
            )
            unwrap_model(model).load_state_dict(state_dict)
            restored_best = True
            if rank == 0:
                print(
                    "[EARLY-STOP] restored best weights from "
                    f"{best_selection_path.name} "
                    f"(step {evaluation_state['early_stopping']['best_step']})",
                    flush=True,
                )
        if config["evaluation"].get("val_full_at_end", True):
            already_full = (
                evaluation_state["last_full_metrics"].get("checkpoint_step")
                == global_step
            )
            if not already_full:
                result = _run_evaluation(
                    kind="val_full_final",
                    model=model,
                    dataloader=loaders.val_full,
                    device=device,
                    config=config,
                    output_dir=output_dir,
                    global_step=global_step,
                    rank=rank,
                    world_size=world_size,
                    total_steps=total_steps,
                    evaluation_state=evaluation_state,
                )
                evaluation_state["full_test_count"] += 1
                evaluation_state["val_full_count"] += 1
                final_metrics = _broadcast_object(
                    result.metrics, rank
                )
                evaluation_state["last_full_metrics"] = final_metrics
                if _is_better_model(
                    final_metrics, best_metrics, config["evaluation"]
                ):
                    best_metrics = final_metrics
                    evaluation_state["best_observed_dev_test_metrics"] = final_metrics
                    evaluation_state["best_validation_metrics"] = final_metrics
                    final_is_best = True
                else:
                    final_is_best = False
        _save_all_ranks(
            output_dir=output_dir,
            tag="last",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            global_step=global_step,
            best_metrics=best_metrics,
            config=config,
            sampler=loaders.sampler,
            evaluation_state=evaluation_state,
            rank=rank,
            world_size=world_size,
            force_full_model=True,
        )
        if (
            final_is_best
            and _save_best_enabled(config["checkpoint"])
            and rank == 0
        ):
            clone_checkpoint_pair(
                output_dir / "checkpoints",
                "last",
                "best_selection",
            )
            clone_checkpoint_pair(
                output_dir / "checkpoints",
                "last",
                "best_observed_dev_test_selection",
            )
        if (
            rank == 0
            and int(config["checkpoint"].get("save_topk", 0)) > 0
            and final_metrics
        ):
            _maybe_save_topk(
                output_dir=output_dir,
                checkpoint_cfg=config["checkpoint"],
                metrics=final_metrics,
                global_step=global_step,
                evaluation_state=evaluation_state,
            )
        if tb_writer is not None:
            tb_writer.close()
        distributed_barrier()
        if rank == 0:
            if frozen_snapshot is not None:
                assert_frozen_parameters_unchanged(frozen_snapshot, model)
                print("Verified: every frozen parameter remained bitwise unchanged.")
            summary_payload = {
                "global_step": global_step,
                "last_checkpoint_metrics": evaluation_state["last_full_metrics"],
                "best_observed_dev_test_metrics": best_metrics,
                "best_validation_metrics": best_metrics,
                "best_val_loss_metrics": best_val_loss_metrics,
                "best_worst_game_metrics": best_worst_game_metrics,
                "restored_best": restored_best,
                "early_stopping": evaluation_state["early_stopping"],
                "topk_checkpoints": evaluation_state.get("topk_registry", []),
                "test_evaluation_counts": {
                    "quick": evaluation_state["quick_test_count"],
                    "full": evaluation_state["full_test_count"],
                    "train_probe": evaluation_state["train_probe_count"],
                },
                "data_pipeline": loaders.data_summary,
                "train_sampling_snapshot": [
                    {
                        "game": game,
                        "label": label,
                        "delta": delta,
                        "count": count,
                    }
                    for (game, label, delta), count in sorted(
                        getattr(
                            loaders.sampler,
                            "last_epoch_game_label_delta_counts",
                            {},
                        ).items()
                    )
                ],
            }
            (output_dir / "training_summary.json").write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _finalize_run_success(
                output_dir=output_dir,
                config=config,
                run_id=run_id,
                run_mode=run_mode,
                runs_root=runs_root,
                summary_payload=summary_payload,
                global_step=global_step,
                total_steps=total_steps,
                best_metrics=best_metrics,
                started_wall=started_wall,
                parent_run_id=(run_meta or {}).get("parent_run_id"),
            )
        distributed_barrier()
        return {
            "global_step": global_step,
            "last_metrics": evaluation_state["last_full_metrics"],
            "best_metrics": best_metrics,
            "evaluation_state": evaluation_state,
            "output_dir": str(output_dir),
            "run_id": run_id,
            "state": STATE_SUCCEEDED,
        }
    except BaseException as exc:
        if rank == 0 and output_dir is not None:
            _record_run_failure(
                output_dir,
                exc,
                global_step=global_step,
                run_id=run_id,
                run_mode=run_mode,
                runs_root=runs_root,
                config=config,
                started_wall=started_wall,
            )
        raise
    finally:
        cleanup_distributed()
