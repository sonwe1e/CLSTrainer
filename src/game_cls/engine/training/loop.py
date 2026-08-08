from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from game_cls.config_schema import (
    finalize_config,
    schedule_budget_warnings,
)
from game_cls.contract import stamp_payload
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.checkpoint import (
    clone_checkpoint_pair,
    restore_random_state,
    restore_training_checkpoint,
    unwrap_model,
)
from game_cls.engine.device import autocast_context
from game_cls.engine.training.config_validation import (
    _dataloader_option,
    validate_training_config,
)
from game_cls.engine.training.early_stopping import (
    _early_stopping_defaults,
    _early_stopping_monitor_label,
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
from game_cls.engine.training.optimizer import _set_train_mode
from game_cls.engine.training.run_io import (
    _append_training_metrics,
    _finalize_run_success,
    _iso_now,
    _maybe_save_topk,
    _read_status,
    _record_run_failure,
    _write_resolved_config,
    _write_run_manifest,
    acquire_resume_lock,
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
    _schedule_factor,
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
from game_cls.runtime.distributed_runtime import (
    barrier as distributed_barrier,
)
from game_cls.runtime.distributed_runtime import (
    cleanup as cleanup_distributed,
)
from game_cls.runtime.distributed_runtime import (
    init_runtime as initialize_runtime,
)
from game_cls.runtime.distributed_runtime import (
    is_initialized as is_distributed,
)


def _transfer_optimizer_state(old_optimizer, new_optimizer) -> None:
    """Copy per-parameter AdamW state from ``old`` into ``new``.

    Parameters present in both keep their momentum/variance/step; params
    newly unfrozen get fresh state (the desired behavior for staged
    unfreeze). State is keyed by the parameter Tensor itself.
    """
    old_state = {
        parameter: state for parameter, state in old_optimizer.state.items() if state
    }
    for group in new_optimizer.param_groups:
        for parameter in group["params"]:
            if parameter in old_state:
                new_optimizer.state[parameter] = old_state[parameter]


def _unfreeze_boundary(rules, global_step: int) -> bool:
    """True when ``global_step`` is exactly a rule's unfreeze_at_step."""
    return any(rule.unfreeze_at_step == global_step for rule in rules)


def _rule_frozen_at_zero(rules, parameter_name: str) -> bool:
    """True when ``parameter_name`` is not trainable at step 0 under rules."""
    from game_cls.model.trainable_rules import rule_for

    rule = rule_for(rules, parameter_name)
    return rule is None or rule.unfreeze_at_step > 0


def _wrap_distributed(bare_model, device, local_rank: int):
    """Wrap ``bare_model`` in DDP with this project's fixed options.

    DDP binds its Reducer buckets to the ``requires_grad`` set that exists at
    construction time. Staged unfreeze therefore has to re-enter this helper
    at every boundary that changes the trainable set, otherwise the freshly
    unfrozen parameters sit in no bucket at all and their gradients are never
    all-reduced -- each rank would silently train its own copy.
    """
    from torch.nn.parallel import DistributedDataParallel

    # CPU DDP must NOT pass device_ids (it has no CUDA devices);
    # only CUDA/NPU rank-to-device binding is valid.
    ddp_kwargs: dict[str, Any] = {}
    if device.type in ("cuda", "npu"):
        ddp_kwargs["device_ids"] = [local_rank]
    return DistributedDataParallel(
        bare_model,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        **ddp_kwargs,
    )


def _rewrap_distributed(bare_model, device, local_rank: int):
    """Build and return a fresh DDP wrapper around ``bare_model``.

    The caller is responsible for releasing the last strong reference to the
    old DDP wrapper and calling gc.collect() + distributed_barrier() BEFORE
    calling this function, so that the old Reducer is provably dead before the
    new one is constructed.  Pattern::

        distributed_barrier()
        _old = model
        model = None
        del _old
        gc.collect()
        model = _rewrap_distributed(bare_model, device, local_rank)

    The ``model`` parameter was removed from this signature (audit P0-8) to
    prevent the Python reference-counting trap where ``del model`` inside the
    function has no effect on the caller's binding.
    """
    return _wrap_distributed(bare_model, device, local_rank)


def _prune_unfrozen_from_snapshot(
    frozen_snapshot: dict | None, bare_model
) -> dict | None:
    """Drop parameters that have since been unfrozen from the frozen snapshot.

    ``verify_frozen_parameters`` asserts that every parameter captured while
    frozen is still bitwise identical at the end of the run. Staged unfreeze
    legitimately starts training some of those parameters, so their entries
    have to leave the snapshot at the unfreeze boundary. Parameters that stay
    frozen keep their ORIGINAL captured value, so the check still covers the
    whole run for them rather than restarting at each boundary.
    """
    if frozen_snapshot is None:
        return None
    trainable_now = {
        name
        for name, parameter in bare_model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value
        for name, value in frozen_snapshot.items()
        if name not in trainable_now
    }


def run_training(
    config: dict[str, Any],
    run_meta: dict[str, Any] | None = None,
    on_run_dir: Callable[[Path], None] | None = None,
) -> dict:
    """Train the dual-frame classifier.

    Fresh training always allocates an immutable timestamped run directory.
    Exact resume writes only into the explicitly selected existing run.

    ``run_meta`` carries launch facts (command, environment, checkpoint
    hash) for the manifest; ``on_run_dir`` is a rank-0 callback fired as
    soon as the run directory exists (used to attach console capture).
    """
    import torch

    config = finalize_config(config)
    validate_training_config(config)
    image_spec = ImageSpec.from_config(config["data"])
    rank, world_size, local_rank, device = initialize_runtime(config)
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
    resume_lock: Path | None = None
    try:
        seed = int(config["experiment"]["seed"])
        _seed_everything(seed + rank)
        config_output = Path(config["experiment"]["output_dir"])
        if not resuming:
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
            output_dir = config_output
            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
        if resuming and rank == 0:
            # Audit acceptance #6: two processes must not resume the same run
            # concurrently (they would both load the same checkpoint and
            # interleave writes). An O_EXCL lock makes the second start fail.
            assert output_dir is not None
            resume_lock = acquire_resume_lock(output_dir)
        if rank == 0:
            _write_resolved_config(output_dir, config)
            effective_run_id = _write_run_manifest(
                output_dir,
                config=config,
                run_meta=run_meta,
                run_id=run_id,
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
        from game_cls.model.freeze_policy import FreezeSummary
        from game_cls.model.trainable_rules import (
            apply_trainable_state,
            parse_rules,
        )

        rules = parse_rules(config["model"]["trainable_rules"])
        if checkpoint_path:
            # Step5 P5: under DDP only rank 0 touches the base checkpoint
            # file; the state dict is broadcast so the other ranks skip the
            # I/O (weights_only=True load, unchanged trust semantics).
            if world_size > 1 and is_distributed():
                if rank == 0:
                    report = load_model_checkpoint(model, checkpoint_path)
                    state_payload = unwrap_model(model).state_dict()
                else:
                    report = None
                    state_payload = None
                state_payload = _broadcast_object(state_payload, rank)
                if rank != 0:
                    unwrap_model(model).load_state_dict(state_payload, strict=True)
            else:
                report = load_model_checkpoint(model, checkpoint_path)
            if (
                not config["data"].get("synthetic", False)
                and config["model"].get("require_pretrained_backbone", True)
                and rank == 0
            ):
                assert report is not None
                frozen_names = {
                    name
                    for name, _ in model.named_parameters()
                    if _rule_frozen_at_zero(rules, name)
                }
                coverage = validate_production_load(
                    model,
                    report,
                    frozen_parameter_names=frozen_names,
                )
            else:
                coverage = None
            if rank == 0:
                assert report is not None
                print(f"Loaded {len(report.loaded)} model tensors")
                if coverage is not None:
                    print(f"Frozen backbone checkpoint coverage: {coverage:.2%}")
                print(f"Missing: {report.missing}")
                print(f"Unexpected: {report.unexpected}")
                print(f"Shape mismatch: {report.shape_mismatch}")
        apply_trainable_state(model, rules, 0)
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_count = sum(
            parameter.numel()
            for _, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        frozen_count = sum(
            parameter.numel()
            for _, parameter in model.named_parameters()
            if not parameter.requires_grad
        )
        summary = FreezeSummary(tuple(trainable_names), trainable_count, frozen_count)
        _set_train_mode(model, config["model"])
        model.to(device)
        # ``bare_model`` always refers to the undecorated module. Every
        # requires_grad / optimizer-group / freeze-snapshot operation must go
        # through it: DDP prefixes parameter names with "module.", which does
        # not match the anchored regexes in ``model.trainable_rules`` nor the
        # unprefixed names stored in checkpoints' ``trainable_state``.
        bare_model = model
        if world_size > 1:
            model = _wrap_distributed(bare_model, device, local_rank)
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
            for warning in schedule_budget_warnings(config):
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
        # The trainable name set the CURRENT optimizer/DDP wrapper was built
        # from. The unfreeze boundary compares against this instead of
        # recomputing step-1, so a rule with unfreeze_at_step == 0 does not
        # trigger a pointless rebuild on the first batch.
        current_trainable: set[str] = {
            name
            for name, parameter in bare_model.named_parameters()
            if parameter.requires_grad
        }
        from game_cls.model.trainable_rules import (
            build_optimizer_parameter_groups as build_rule_groups,
        )

        optimizer = torch.optim.AdamW(
            build_rule_groups(
                bare_model,
                rules,
                step=0,
                weight_decay=float(config["optimizer"]["weight_decay"]),
                base_lr=float(config["optimizer"]["learning_rate"]),
            )
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
            "val_quick_count": 0,
            "val_full_count": 0,
            "train_probe_count": 0,
            "last_full_metrics": {},
            "last_train_probe_metrics": {},
            "best_validation_metrics": {},
            "best_val_loss_metrics": {},
            "best_worst_game_metrics": {},
            "early_stopping": _early_stopping_defaults(),
            "train_delta_history": [],
            "topk_registry": [],
        }
        resume_path = train_cfg.get("resume_path")
        if resume_path:
            # Step5 P4 rule-aware resume: peek the checkpoint to learn which
            # step it was saved at, re-apply the rules for THAT step, and
            # rebuild the optimizer/scheduler BEFORE restoring. The saved
            # optimizer state has one group per (rule, decay) pair that was
            # live at save time, so a checkpoint taken past an unfreeze
            # boundary has more groups than the step-0 optimizer built above;
            # loading into that optimizer raises "loaded state dict has a
            # different number of parameter groups".
            if rules:
                import torch

                from game_cls.model.trainable_rules import (
                    apply_trainable_state as apply_rules_at_step,
                )
                from game_cls.model.trainable_rules import (
                    build_optimizer_parameter_groups as build_rule_groups,
                )
                from game_cls.model.trainable_rules import rules_fingerprint

                peek = torch.load(resume_path, map_location="cpu", weights_only=True)
                saved_fingerprint = peek.get("trainable_rules_fingerprint")
                if saved_fingerprint and saved_fingerprint != rules_fingerprint(rules):
                    raise RuntimeError(
                        "Resume blocked: model.trainable_rules changed since "
                        "this checkpoint (rules fingerprint differs). Use "
                        "--fork to start a new run from an old checkpoint."
                    )
                saved_step = int(peek.get("global_step", 0))
                # ``trainable_state`` was written from unwrap_model(...), so it
                # holds unprefixed names -- compare against bare_model.
                saved_trainable = set(peek.get("trainable_state") or [])
                apply_rules_at_step(bare_model, rules, saved_step)
                current_trainable = {
                    name
                    for name, parameter in bare_model.named_parameters()
                    if parameter.requires_grad
                }
                if saved_trainable and current_trainable != saved_trainable:
                    raise RuntimeError(
                        "Resume blocked: replaying model.trainable_rules at the "
                        f"saved step ({saved_step}) does not reproduce the saved "
                        "requires_grad mask: "
                        f"missing={sorted(saved_trainable - current_trainable)}, "
                        f"unexpected={sorted(current_trainable - saved_trainable)}. "
                        "Use --fork to start a new run from this checkpoint."
                    )
                # The wrapper built above froze its Reducer around the step-0
                # trainable set; the restored set is generally larger, so DDP
                # has to be rebuilt before the first backward pass. No forward
                # has run yet, so there are no gradient bucket-view aliases to
                # release here.  Release the old wrapper reference in caller
                # scope so gc.collect() can actually free it (audit P0-8).
                if world_size > 1:
                    distributed_barrier()
                    _old_model = model
                    model = None
                    del _old_model
                    gc.collect()
                    model = _rewrap_distributed(bare_model, device, local_rank)
                optimizer = torch.optim.AdamW(
                    build_rule_groups(
                        bare_model,
                        rules,
                        step=saved_step,
                        weight_decay=float(config["optimizer"]["weight_decay"]),
                        base_lr=float(config["optimizer"]["learning_rate"]),
                    )
                )
                scheduler = _build_scheduler(
                    optimizer, config["scheduler"], total_steps
                )
            checkpoint = restore_training_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                scaler,
                expected_base_checkpoint=config["model"].get("checkpoint_path"),
                current_config=config,
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
            frozen_snapshot = snapshot_frozen_parameters(bare_model)
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
                if _unfreeze_boundary(rules, global_step):
                    # Step5 P4: a rule crossed its unfreeze_at_step — make the
                    # newly unfrozen parameters trainable, rebuild the
                    # optimizer (preserving state for already-trainable
                    # params), and resync the scheduler.
                    from game_cls.model.trainable_rules import (
                        apply_trainable_state as apply_rules_at_step,
                    )
                    from game_cls.model.trainable_rules import (
                        build_optimizer_parameter_groups as build_rule_groups,
                    )
                    from game_cls.model.trainable_rules import (
                        resolve_trainable_names,
                    )

                    # _unfreeze_boundary is only a cheap pre-filter: a rule with
                    # unfreeze_at_step == 0 matches on the very first batch even
                    # though the trainable set was already applied before the
                    # optimizer was built. Rebuild only when the set actually
                    # moves relative to what the current optimizer was built
                    # from. The decision is a pure function of (rules,
                    # global_step, parameter names), so every rank reaches the
                    # same verdict and the DDP re-wrap below stays collective.
                    after_names, _ = resolve_trainable_names(
                        bare_model, rules, global_step
                    )
                    if set(after_names) != current_trainable:
                        apply_rules_at_step(bare_model, rules, global_step)
                        _set_train_mode(model, config["model"])
                        if world_size > 1:
                            # Release the gradients that alias the old
                            # Reducer's bucket views before it is discarded.
                            optimizer.zero_grad(set_to_none=True)
                            distributed_barrier()
                            _old_model = model
                            model = None
                            del _old_model
                            gc.collect()
                            model = _rewrap_distributed(bare_model, device, local_rank)
                            _set_train_mode(model, config["model"])
                        old_optimizer = optimizer
                        optimizer = torch.optim.AdamW(
                            build_rule_groups(
                                bare_model,
                                rules,
                                step=global_step,
                                weight_decay=float(config["optimizer"]["weight_decay"]),
                                base_lr=float(config["optimizer"]["learning_rate"]),
                            )
                        )
                        _transfer_optimizer_state(old_optimizer, optimizer)
                        scheduler = _build_scheduler(
                            optimizer, config["scheduler"], total_steps
                        )
                        # Audit P1-3: the fresh LambdaLR starts every rebuilt
                        # group at its pre-decay LR and only recording
                        # ``last_epoch`` does not recompute the actual group
                        # LR, so the first optimizer step after the boundary
                        # would use factor(0) -- a transient jump off the
                        # continuous schedule that is catastrophic inside
                        # warmup. Position every group at the schedule's value
                        # for this step without an out-of-order
                        # ``scheduler.step()``.
                        _base_lr = max(
                            group["initial_lr"] for group in optimizer.param_groups
                        )
                        _factor = _schedule_factor(
                            config["scheduler"], total_steps, _base_lr, global_step
                        )
                        scheduler.last_epoch = global_step
                        for group in optimizer.param_groups:
                            group["lr"] = group["initial_lr"] * _factor
                        # Newly unfrozen parameters are allowed to change from
                        # here on, so they must leave the frozen snapshot.
                        frozen_snapshot = _prune_unfrozen_from_snapshot(
                            frozen_snapshot, bare_model
                        )
                        newly = sorted(set(after_names) - current_trainable)
                        current_trainable = set(after_names)
                        if rank == 0:
                            print(
                                f"[TRAINABLE-RULES] step={global_step}: "
                                f"unfroze {len(newly)} parameter tensors; "
                                f"optimizer rebuilt with "
                                f"{len(optimizer.param_groups)} groups"
                                + (
                                    "; DDP re-wrapped so the new gradients "
                                    "are all-reduced."
                                    if world_size > 1
                                    else "."
                                ),
                                flush=True,
                            )
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
                    loss_cfg = {
                        **config["loss"],
                        "threshold": config["decision"]["threshold"],
                    }
                    loss, components = combined_loss(
                        logits, labels, loss_cfg, global_step, total_steps
                    )
                timing["host_forward_enqueue"] += time.perf_counter() - forward_started
                backward_started = time.perf_counter()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in bare_model.parameters() if p.requires_grad],
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
                            config["evaluation"],
                        ):
                            early_state["stop_reason"] = "validation_plateau_quick"
                            early_state["stopped_at_step"] = global_step
                            early_stop_requested = True
                            if rank == 0:
                                print(
                                    "[EARLY-STOP] quick validation plateau: "
                                    f"monitor={_early_stopping_monitor_label(early_cfg)} "
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
                            config["evaluation"],
                        ):
                            early_state["stop_reason"] = "validation_plateau"
                            early_state["stopped_at_step"] = global_step
                            early_stop_requested = True
                            if rank == 0:
                                print(
                                    "[EARLY-STOP] validation plateau: "
                                    f"monitor={_early_stopping_monitor_label(early_cfg)} "
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
                                evaluation_cfg=config["evaluation"],
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
                            # Raw last-batch values retained for per-batch
                            # diagnostics; noisy by nature.
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
                            "interval_positive_recall_at_decision_threshold": (
                                interval_metrics[
                                    "interval_positive_recall_at_decision_threshold"
                                ]
                            ),
                            "interval_negative_specificity_at_decision_threshold": (
                                interval_metrics[
                                    "interval_negative_specificity_at_decision_threshold"
                                ]
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
        final_metrics: dict = {}
        early_cfg = config.get("early_stopping") or {}
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
                evaluation_state["val_full_count"] += 1
                final_metrics = _broadcast_object(result.metrics, rank)
                evaluation_state["last_full_metrics"] = final_metrics
                if _is_better_model(final_metrics, best_metrics, config["evaluation"]):
                    best_metrics = final_metrics
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
                evaluation_cfg=config["evaluation"],
            )
        # Audit P0-1: checkpoint_last must be the true terminal training state.
        # The save above ran before any weight swap, so its model / optimizer /
        # scheduler / sampler / RNG / global_step are mutually consistent and
        # ``--resume checkpoint_last`` is an exact resume. ``restore_best`` is
        # a deployment/display concern only: the selected weights are loaded
        # AFTER the lineage is finalized so they can never leak into ``last``.
        restored_best = False
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
        if tb_writer is not None:
            tb_writer.close()
        distributed_barrier()
        if rank == 0:
            if frozen_snapshot is not None:
                assert_frozen_parameters_unchanged(frozen_snapshot, bare_model)
                print("Verified: every frozen parameter remained bitwise unchanged.")
            summary_payload = {
                "global_step": global_step,
                "last_checkpoint_metrics": evaluation_state["last_full_metrics"],
                "best_validation_metrics": best_metrics,
                "best_val_loss_metrics": best_val_loss_metrics,
                "best_worst_game_metrics": best_worst_game_metrics,
                "restored_best": restored_best,
                "early_stopping": evaluation_state["early_stopping"],
                "topk_checkpoints": evaluation_state.get("topk_registry", []),
                "validation_evaluation_counts": {
                    "quick": evaluation_state["val_quick_count"],
                    "full": evaluation_state["val_full_count"],
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
                json.dumps(
                    stamp_payload(summary_payload), ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            _finalize_run_success(
                output_dir=output_dir,
                config=config,
                run_id=run_id,
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
                runs_root=runs_root,
                config=config,
                started_wall=started_wall,
            )
        raise
    finally:
        if resume_lock is not None:
            resume_lock.unlink(missing_ok=True)
        cleanup_distributed()
