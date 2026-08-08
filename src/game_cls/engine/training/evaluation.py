from __future__ import annotations

from pathlib import Path
from typing import Any

from game_cls.engine.checkpoint import (
    unwrap_model,
)
from game_cls.engine.evaluator import EvaluationOutput, evaluate
from game_cls.engine.training.run_io import (
    _EVALUATION_ROLES,
    _EVALUATION_SCOPES,
    _append_evaluation_history,
    _evaluation_history_record,
)
from game_cls.engine.training.selection import _annotate_selection
from game_cls.losses.threshold_loss import (
    threshold_weight_at_step,
    threshold_weight_from_steps,
)
from game_cls.reports.error_writer import (
    prepare_evaluation_directory,
    write_evaluation_report,
)
from game_cls.runtime.distributed_runtime import (
    barrier as distributed_barrier,
)
from game_cls.runtime.distributed_runtime import (
    is_initialized as is_distributed,
)


def _threshold_weight_for_eval(config: dict, step: int, total_steps: int) -> float:
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


def _new_interval_accumulator(device) -> dict[str, Any]:
    """Sample-weighted interval accumulators (device tensors, no sync).

    Accumulators are float32: NPU/CANN does not support float64 device
    tensors (``k::double`` errors), and the sample-weighted means only
    need float32 precision. Do not switch these back to float64.
    """
    import torch

    return {
        "loss_sum": torch.zeros((), dtype=torch.float32, device=device),
        "ce_sum": torch.zeros((), dtype=torch.float32, device=device),
        "threshold_sum": torch.zeros((), dtype=torch.float32, device=device),
        "threshold_weight_sum": torch.zeros((), dtype=torch.float32, device=device),
        "tail_sum": torch.zeros((), dtype=torch.float32, device=device),
        "rank_sum": torch.zeros((), dtype=torch.float32, device=device),
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
            accum["threshold_weight_sum"],
            accum["tail_sum"],
            accum["rank_sum"],
        )
    )
    samples = torch.tensor([accum["samples"]], dtype=torch.float32, device=device)
    counts = accum["counts"].clone()
    if is_distributed():
        import torch.distributed as dist

        dist.all_reduce(floating)
        dist.all_reduce(samples)
        dist.all_reduce(counts)
    sample_count = float(samples.item())
    (
        loss_sum,
        ce_sum,
        threshold_sum,
        threshold_weight_sum,
        tail_sum,
        rank_sum,
    ) = floating.tolist()
    tp, fp, fn, tn = counts.tolist()
    positive_total = tp + fn
    negative_total = fp + tn
    denominator = max(sample_count, 1.0)
    return {
        "interval_loss": loss_sum / denominator,
        "interval_ce": ce_sum / denominator,
        "interval_threshold_loss": threshold_sum / denominator,
        "interval_threshold_weight": threshold_weight_sum / denominator,
        "interval_negative_tail_loss": tail_sum / denominator,
        "interval_rank_loss": rank_sum / denominator,
        "interval_accuracy": (tp + tn) / max(sample_count, 1.0)
        if sample_count
        else 0.0,
        "interval_positive_recall_at_decision_threshold": (
            tp / positive_total if positive_total else None
        ),
        "interval_negative_specificity_at_decision_threshold": (
            tn / negative_total if negative_total else None
        ),
        "interval_samples": int(sample_count),
    }


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
        config["decision"]["threshold"],
        checkpoint_step=global_step,
        distributed=is_distributed(),
        rank=rank,
        world_size=world_size,
        evaluation_kind=("quick" if scope == "quick" else kind),
        report_dir=report_dir,
        full_auc_mode=config["evaluation"].get("full_auc_mode", "histogram"),
        auc_histogram_bins=int(config["evaluation"].get("auc_histogram_bins", 4096)),
        quick_error_limit=int(config["evaluation"].get("quick_save_error_limit", 200)),
        amp=bool(config["evaluation"].get("amp", config["device"].get("amp", False))),
        amp_dtype=str(
            config["evaluation"].get(
                "amp_dtype",
                config["device"].get("amp_dtype", "bfloat16"),
            )
        ),
        parquet_row_group_size=int(
            config["evaluation"].get("parquet_row_group_size", 4096)
        ),
        group_catalogs=getattr(
            getattr(dataloader, "dataset", None),
            "group_catalogs",
            None,
        ),
        threshold_loss_weight=threshold_weight,
        cross_entropy_weight=float(loss_cfg.get("cross_entropy_weight", 1.0)),
        threshold_safety_margin=float(loss_cfg.get("threshold_safety_margin", 0.20)),
        threshold_temperature=float(loss_cfg.get("threshold_temperature", 0.50)),
        max_fpr_for_recall=float(config["evaluation"].get("max_fpr_for_recall", 0.01)),
        tail_calibration_enabled=bool(
            config["evaluation"].get("tail_calibration_enabled", True)
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
                    config["evaluation"].get("amp", config["device"].get("amp", False))
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
            probe_metrics = evaluation_state.get("last_train_probe_metrics") or {}
            if probe_metrics.get("checkpoint_step") == global_step:
                probe_ce = probe_metrics.get("cross_entropy")
                val_ce = metrics.get("cross_entropy")
                probe_score = probe_metrics.get("selection_score")
                val_score = metrics.get("selection_score")
                if isinstance(probe_ce, (int, float)) and isinstance(
                    val_ce, (int, float)
                ):
                    history_extra["generalization_ce_gap"] = float(val_ce) - float(
                        probe_ce
                    )
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
