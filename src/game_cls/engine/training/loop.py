from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from game_cls.config_schema import finalize_config, split_role_warnings
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.checkpoint import (
    clone_checkpoint_pair,
    restore_random_state,
    restore_training_checkpoint,
    unwrap_model,
)
from game_cls.engine.device import autocast_context
from game_cls.engine.distributed import (
    cleanup_distributed,
    distributed_barrier,
    initialize_runtime,
)
from game_cls.engine.training.config_validation import (
    _dataloader_option,
    validate_training_config,
)
from game_cls.engine.training.early_stopping import (
    _early_stopping_defaults,
    _update_early_stopping,
)
from game_cls.engine.training.evaluation import (
    _new_interval_accumulator,
    _reduce_interval_accumulator,
    _run_evaluation,
)
from game_cls.engine.training.loaders import _make_dataloaders
from game_cls.engine.training.loop_util import (
    _seed_everything,
    _synchronize_device_for_metrics,
    _tb_write_scalars,
)
from game_cls.engine.training.optimizer import (
    _set_train_mode,
    build_optimizer_parameter_groups,
)
from game_cls.engine.training.run_io import (
    _append_training_metrics,
    _finalize_run_success,
    _iso_now,
    _maybe_save_topk,
    _read_status,
    _record_run_failure,
    _write_resolved_config,
    _write_run_manifest,
)
from game_cls.engine.training.selection import (
    _is_better_model,
    _metric_value,
    _save_best_enabled,
)
from game_cls.engine.training.state import (
    _broadcast_object,
    _build_scheduler,
    _save_all_ranks,
)
from game_cls.losses.threshold_loss import (
    combined_loss,
    probability_threshold_to_margin,
)
from game_cls.model.builder import build_model
from game_cls.model.checkpoint_loader import (
    load_model_checkpoint,
    validate_production_load,
)
from game_cls.model.freeze_policy import (
    assert_frozen_parameters_unchanged,
    configure_trainable_parameters,
    snapshot_frozen_parameters,
)
from game_cls.runs import (
    STATE_RUNNING,
    STATE_SUCCEEDED,
    allocate_run_dir,
    append_resume_event,
    append_run_index,
    atomic_write_text,
    render_overview_html,
    update_status,
)


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
    resuming = bool(config["train"].get("resume_path") or meta.get("resumed_from"))
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
            print(
                "Data pipeline:", json.dumps(loaders.data_summary, ensure_ascii=False)
            )
            train_workers = int(_dataloader_option(config, "train", "num_workers", 0))
            eval_workers = int(_dataloader_option(config, "eval", "num_workers", 0))
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
            timeout = _dataloader_option(config, "train", "timeout_seconds", 180)
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
        evaluation_state: dict[str, Any] = {
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
                sampler_state.get("step_in_epoch", checkpoint.get("step_in_epoch", 0))
            )
            best_metrics = dict(checkpoint.get("best_metrics", {}))
            evaluation_state.update(checkpoint.get("evaluation_state", {}))
            # Old checkpoints predate the early-stopping state; resume must
            # not recompute patience from scratch for the restored run.
            evaluation_state.setdefault("early_stopping", _early_stopping_defaults())
            evaluation_state.setdefault("last_train_probe_metrics", {})
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
        if rank == 0 and bool(config["evaluation"].get("tensorboard_live", False)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
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
                timing["host_h2d_enqueue"] += time.perf_counter() - transfer_started
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
                timing["host_forward_enqueue"] += time.perf_counter() - forward_started
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
                step_samples = int(train_cfg["local_batch_size"]) * world_size
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
                        logits.detach().float()[:, 1] - logits.detach().float()[:, 0]
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
                    loss.detach()
                    .to(interval_accum["loss_sum"].dtype)
                    .mul_(batch_samples)
                )
                interval_accum["ce_sum"].add_(
                    components["cross_entropy"]
                    .to(interval_accum["ce_sum"].dtype)
                    .mul_(batch_samples)
                )
                interval_accum["threshold_sum"].add_(
                    components["threshold_loss"]
                    .to(interval_accum["threshold_sum"].dtype)
                    .mul_(batch_samples)
                )
                interval_accum["threshold_weight_sum"].add_(
                    float(components["threshold_weight"]) * batch_samples
                )
                interval_accum["tail_sum"].add_(
                    torch.as_tensor(components.get("negative_tail_loss", 0.0))
                    .to(interval_accum["tail_sum"].dtype)
                    .mul_(batch_samples)
                )
                interval_accum["rank_sum"].add_(
                    torch.as_tensor(components.get("rank_loss", 0.0))
                    .to(interval_accum["rank_sum"].dtype)
                    .mul_(batch_samples)
                )
                interval_accum["samples"] += batch_samples

                log_every = int(train_cfg["log_every_steps"])

                evaluation_cfg = config["evaluation"]
                probe_every = int(evaluation_cfg.get("train_probe_every_steps", 0))
                quick_every = int(evaluation_cfg.get("val_quick_every_steps", 0))
                full_every = int(evaluation_cfg.get("val_full_every_steps", 0))
                run_full = bool(full_every and global_step % full_every == 0)
                run_probe = bool(
                    probe_every
                    and global_step % probe_every == 0
                    and loaders.train_probe is not None
                )
                run_quick = bool(
                    quick_every and global_step % quick_every == 0 and not run_full
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
                    evaluation_seconds += time.perf_counter() - evaluation_started
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
                    if early_cfg.get("enabled", False) and not early_cfg.get(
                        "full_validation_only", True
                    ):
                        quick_metrics = _broadcast_object(result.metrics, rank)
                        early_state = evaluation_state["early_stopping"]
                        if _update_early_stopping(
                            early_state,
                            early_cfg,
                            quick_metrics or {},
                            global_step,
                        ):
                            early_state["stop_reason"] = "validation_plateau_quick"
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
                    evaluation_seconds += time.perf_counter() - evaluation_started

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
                        < float(best_val_loss_metrics.get("cross_entropy") or 0.0)
                    )
                    if loss_improved:
                        best_val_loss_metrics = metrics
                        evaluation_state["best_val_loss_metrics"] = metrics
                    worst_f1 = (
                        _metric_value(metrics, "worst_game_f1_at_decision_threshold")
                        if isinstance(metrics, dict)
                        else None
                    )
                    worst_improved = isinstance(worst_f1, (int, float)) and (
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
                        evaluation_state["best_worst_game_metrics"] = metrics
                    early_cfg = config.get("early_stopping") or {}
                    if early_cfg.get("enabled", False):
                        early_state = evaluation_state["early_stopping"]
                        if _update_early_stopping(
                            early_state,
                            early_cfg,
                            metrics or {},
                            global_step,
                        ):
                            early_state["stop_reason"] = "validation_plateau"
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
                                duration_seconds=(time.time() - started_wall),
                                config=config,
                                summary_payload=None,
                            )
                            atomic_write_text(
                                output_dir / "overview.html", overview_html
                            )
                        except OSError:
                            pass
                    _set_train_mode(model, config["model"])
                    evaluation_seconds += time.perf_counter() - evaluation_started

                save_every = int(config["checkpoint"].get("save_last_every_steps", 0))
                save_last_due = bool(save_every and global_step % save_every == 0)
                any_improved = is_best or loss_improved or worst_improved
                checkpoint_started = None
                topk_requested = bool(
                    run_full and int(config["checkpoint"].get("save_topk", 0)) > 0
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
                    checkpoint_seconds += time.perf_counter() - checkpoint_started

                should_log = (
                    global_step % log_every == 0 or global_step >= run_until_step
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
                            key: value / interval_steps for key, value in timing.items()
                        }
                        loss_value = float(loss.detach().item())
                        ce_value = float(components["cross_entropy"].item())
                        threshold_loss_value = float(
                            components["threshold_loss"].item()
                        )
                        grad_norm_value = float(grad_norm.detach().item())
                        learning_rate = max(
                            float(group["lr"]) for group in optimizer.param_groups
                        )
                        interval_samples_per_second = log_interval_samples / max(
                            interval_seconds, 1e-9
                        )
                        interval_step_time = interval_seconds / interval_steps
                        data_wait_seconds = timing["host_data_wait"]
                        data_wait_ratio = data_wait_seconds / max(
                            interval_seconds, 1e-9
                        )
                        wall_samples_per_second = processed_samples / max(elapsed, 1e-9)
                        metrics_payload = {
                            "step": global_step,
                            "total_steps": total_steps,
                            # Raw last-batch values (kept for backward
                            # compatibility; noisy by nature).
                            "loss": loss_value,
                            "ce": ce_value,
                            "threshold_loss": threshold_loss_value,
                            "threshold_weight": float(components["threshold_weight"]),
                            # Exact sample-weighted interval averages,
                            # distributed-reduced: the canonical curves.
                            "interval_loss": interval_metrics["interval_loss"],
                            "interval_ce": interval_metrics["interval_ce"],
                            "interval_threshold_loss": interval_metrics[
                                "interval_threshold_loss"
                            ],
                            "interval_threshold_weight": interval_metrics[
                                "interval_threshold_weight"
                            ],
                            "interval_negative_tail_loss": interval_metrics[
                                "interval_negative_tail_loss"
                            ],
                            "interval_rank_loss": interval_metrics[
                                "interval_rank_loss"
                            ],
                            "interval_accuracy": interval_metrics["interval_accuracy"],
                            "interval_positive_recall_tau099": (
                                interval_metrics["interval_positive_recall_tau099"]
                            ),
                            "interval_negative_specificity_tau099": (
                                interval_metrics["interval_negative_specificity_tau099"]
                            ),
                            "interval_samples": interval_metrics["interval_samples"],
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
                            "wall_samples_per_second": (wall_samples_per_second),
                            "host_enqueue_timing": averages,
                        }
                        _append_training_metrics(metrics_path, metrics_payload)
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
                            f"negative_tail_loss="
                            f"{interval_metrics['interval_negative_tail_loss']:.6f} "
                            f"rank_loss="
                            f"{interval_metrics['interval_rank_loss']:.6f} "
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
                delta_counts = getattr(loaders.sampler, "last_epoch_delta_counts", None)
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
                        ).append({"epoch": epoch, "failures": int(dedup_failures)})
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
        best_selection_path = output_dir / "checkpoints" / "model_best_selection.pth"
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
                final_metrics = _broadcast_object(result.metrics, rank)
                evaluation_state["last_full_metrics"] = final_metrics
                if _is_better_model(final_metrics, best_metrics, config["evaluation"]):
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
        if final_is_best and _save_best_enabled(config["checkpoint"]) and rank == 0:
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
