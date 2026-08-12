from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from .checkpoint import save_model_weights, save_training_checkpoint
from .data import DistributedEvalSampler, build_datasets
from .distributed import Runtime, barrier, seed_everything, wrap_ddp
from .losses import FocalLoss
from .metrics import BinaryAccumulator, DiagnosticAccumulator
from .models import build_model, load_initial_weights
from .plotting import (
    plot_diagnostics,
    plot_history,
    write_history,
    write_metric_details,
)
from .sampler import BalancedVideoSampler


@dataclass
class LoaderBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_sampler: Sampler[int] | None


def _autocast_context(runtime: Runtime, runtime_cfg: dict[str, Any]):
    enabled = bool(runtime_cfg.get("amp", False)) and runtime.device.type != "cpu"
    if not enabled:
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(runtime_cfg.get("amp_dtype", "bfloat16"))]
    return torch.autocast(device_type=runtime.device.type, dtype=dtype)


def _build_scaler(runtime: Runtime, runtime_cfg: dict[str, Any]):
    use_float16 = bool(runtime_cfg.get("amp", False)) and str(
        runtime_cfg.get("amp_dtype", "bfloat16")
    ) == "float16"
    enabled = use_float16 and runtime.device.type != "cpu"
    try:
        return torch.amp.GradScaler(runtime.device.type, enabled=enabled)
    except (TypeError, ValueError):
        return torch.cuda.amp.GradScaler(enabled=enabled and runtime.device.type == "cuda")


def _loader_options(train_cfg: dict[str, Any], runtime: Runtime) -> dict[str, Any]:
    workers = int(train_cfg["num_workers"])
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(train_cfg.get("pin_memory", False)),
    }
    if workers > 0:
        options["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        options["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        if runtime.device.type == "npu":
            options["multiprocessing_context"] = "spawn"
    return options


def build_dataloaders(config: dict[str, Any], runtime: Runtime) -> LoaderBundle:
    train_cfg = config["train"]
    train_dataset, val_dataset, test_dataset = build_datasets(config)

    train_sampler: Sampler[int] | None = None
    val_sampler = None
    test_sampler = None
    sampler_cfg = config.get("sampler", {})
    if bool(sampler_cfg.get("enabled", False)):
        train_sampler = BalancedVideoSampler(
            train_dataset,
            class_probability=sampler_cfg.get("class_probability", [0.5, 0.5]),
            game_balance_alpha=float(sampler_cfg.get("game_balance_alpha", 0.5)),
            samples_per_epoch=sampler_cfg.get("samples_per_epoch"),
            seed=int(config["experiment"]["seed"]),
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
    elif runtime.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=int(config["experiment"]["seed"]),
            drop_last=False,
        )
    if runtime.distributed:
        val_sampler = DistributedEvalSampler(val_dataset, runtime.rank, runtime.world_size)
        test_sampler = DistributedEvalSampler(test_dataset, runtime.rank, runtime.world_size)

    common = _loader_options(train_cfg, runtime)
    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        **common,
    )
    return LoaderBundle(
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
    )


def _prepare_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True).float().div_(255.0)
    labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
    return images, labels


def _forward_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    runtime: Runtime,
    runtime_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    with _autocast_context(runtime, runtime_cfg):
        logits = model(images[:, 0], images[:, 1])
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(f"Model must return [B,2] logits, got {tuple(logits.shape)}")
        loss = criterion(logits.float(), labels)
    return logits, loss


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    runtime: Runtime,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    diagnostics_cfg = config.get("diagnostics", {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True))
    if diagnostics_enabled:
        games = [str(video.game) for video in loader.dataset.videos]
        accumulator = DiagnosticAccumulator(
            runtime.device,
            games,
            bins=int(diagnostics_cfg.get("histogram_bins", 20)),
        )
    else:
        accumulator = BinaryAccumulator(runtime.device)

    threshold = float(config["train"]["decision_threshold"])
    for batch in loader:
        images, labels = _prepare_batch(batch, runtime.device)
        logits, loss = _forward_loss(
            model, images, labels, criterion, runtime, config["runtime"]
        )
        if diagnostics_enabled:
            accumulator.update(loss, logits, labels, batch["game"], threshold)
        else:
            accumulator.update(loss, logits, labels, threshold)
    accumulator.reduce()
    return accumulator.compute()


def _create_run_dir(config: dict[str, Any], runtime: Runtime) -> Path:
    root = Path(config["experiment"]["output_dir"])
    name = str(config["experiment"].get("name", "run"))
    if runtime.distributed:
        import torch.distributed as dist

        # Use a device tensor instead of broadcast_object_list. This keeps the
        # synchronization path on ordinary HCCL/NCCL/Gloo tensor collectives.
        stamp_ns = time.time_ns() if runtime.is_main else 0
        value = torch.tensor([stamp_ns], dtype=torch.int64, device=runtime.device)
        dist.broadcast(value, src=0)
        stamp_ns = int(value.item())
        stamp = datetime.fromtimestamp(stamp_ns / 1_000_000_000).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    run_dir = root / f"{stamp}_{name}"
    if runtime.is_main:
        run_dir.mkdir(parents=True, exist_ok=False)
    barrier()
    return run_dir


def _build_scheduler(optimizer, train_cfg: dict[str, Any], epochs: int):
    if str(train_cfg.get("scheduler", "cosine")) == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(train_cfg.get("min_learning_rate", 1e-5)),
    )


def _empty_history_row(step: int, epoch: int, learning_rate: float) -> dict[str, Any]:
    return {
        "step": int(step),
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "train_loss": None,
        "train_f1": None,
        "train_samples": None,
        "val_loss": None,
        "val_f1": None,
        "val_samples": None,
        "val_class0_f1": None,
        "val_class1_f1": None,
        "val_class0_recall": None,
        "val_class1_recall": None,
        "test_loss": None,
        "test_f1": None,
        "test_samples": None,
        "test_class0_f1": None,
        "test_class1_f1": None,
        "test_class0_recall": None,
        "test_class1_recall": None,
    }


def _put_metrics(row: dict[str, Any], split: str, metrics: dict[str, Any]) -> None:
    row[f"{split}_loss"] = float(metrics["loss"])
    row[f"{split}_f1"] = float(metrics["f1"])
    row[f"{split}_samples"] = int(metrics["samples"])
    per_class = metrics.get("per_class")
    if per_class is not None and split in {"val", "test"}:
        for class_id in (0, 1):
            current = per_class[str(class_id)]
            row[f"{split}_class{class_id}_f1"] = float(current["f1"])
            row[f"{split}_class{class_id}_recall"] = float(current["recall"])


def _format_metrics(row: dict[str, Any]) -> str:
    fields = [f"step={row['step']}", f"epoch={row['epoch']}"]
    for split in ("train", "val", "test"):
        if row[f"{split}_loss"] is not None:
            fields.append(f"{split}_loss={row[f'{split}_loss']:.6f}")
            fields.append(f"{split}_f1={row[f'{split}_f1']:.4f}")
            if split in {"val", "test"} and row.get(f"{split}_class0_f1") is not None:
                fields.append(f"{split}_c0_f1={row[f'{split}_class0_f1']:.4f}")
                fields.append(f"{split}_c1_f1={row[f'{split}_class1_f1']:.4f}")
    fields.append(f"lr={row['learning_rate']:.3e}")
    return " ".join(fields)


def _build_criterion(config: dict[str, Any], device: torch.device) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = str(loss_cfg.get("type", "cross_entropy"))
    if loss_type == "cross_entropy":
        alpha = loss_cfg.get("alpha")
        weight = None if alpha is None else torch.tensor(alpha, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight)
    if loss_type == "focal":
        return FocalLoss(
            gamma=float(loss_cfg.get("gamma", 1.5)),
            alpha=loss_cfg.get("alpha"),
        ).to(device)
    raise ValueError(f"Unsupported loss type: {loss_type}")


def train(config: dict[str, Any], runtime: Runtime) -> dict[str, Any]:
    seed_everything(int(config["experiment"]["seed"]), runtime.rank)
    run_dir = _create_run_dir(config, runtime)

    loaders = build_dataloaders(config, runtime)
    model = build_model(config["model"])
    load_initial_weights(model, config["model"].get("init_weights"))
    model.to(runtime.device)
    model = wrap_ddp(
        model,
        runtime,
        find_unused_parameters=bool(config["runtime"].get("find_unused_parameters", True)),
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("Model has no trainable parameters. Set requires_grad=True in your model factory.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(config["train"]["epochs"])
    scheduler = _build_scheduler(optimizer, config["train"], epochs)
    scaler = _build_scaler(runtime, config["runtime"])
    criterion = _build_criterion(config, runtime.device)
    train_cfg = config["train"]
    threshold = float(train_cfg["decision_threshold"])
    grad_clip = float(train_cfg.get("gradient_clip_norm", 0.0) or 0.0)
    log_every = int(train_cfg.get("log_every_steps", 0))
    n_val_step = int(train_cfg["n_val_step"])
    n_test_step = int(train_cfg["n_test_step"])

    if runtime.is_main:
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("run_dir:", run_dir)
        print("train:", json.dumps(loaders.train_dataset.summary(), ensure_ascii=False))
        print("val  :", json.dumps(loaders.val_dataset.summary(), ensure_ascii=False))
        print("test :", json.dumps(loaders.test_dataset.summary(), ensure_ascii=False))
        print(
            f"runtime: device={runtime.device} world_size={runtime.world_size} "
            f"backend={runtime.backend or 'none'} amp={config['runtime']['amp']}"
        )
        print(f"evaluation: val_every={n_val_step} steps test_every={n_test_step} steps")
        if isinstance(loaders.train_sampler, BalancedVideoSampler):
            print("sampler:", json.dumps(loaders.train_sampler.summary(), ensure_ascii=False))

    history: list[dict[str, Any]] = []
    metric_details: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_val_loss = math.inf
    best_val_step = 0
    best_val_metrics: dict[str, Any] | None = None
    global_step = 0
    started = time.time()
    train_window = BinaryAccumulator(runtime.device)

    for epoch in range(1, epochs + 1):
        if loaders.train_sampler is not None and hasattr(loaders.train_sampler, "set_epoch"):
            loaders.train_sampler.set_epoch(epoch)
        model.train()

        for batch_index, batch in enumerate(loaders.train_loader, start=1):
            global_step += 1
            images, labels = _prepare_batch(batch, runtime.device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = _forward_loss(
                model, images, labels, criterion, runtime, config["runtime"]
            )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_window.update(loss, logits, labels, threshold)

            epoch_end = batch_index == len(loaders.train_loader)
            final_step = epoch == epochs and epoch_end
            val_due = global_step % n_val_step == 0 or final_step
            test_due = global_step % n_test_step == 0 or final_step
            log_due = log_every > 0 and global_step % log_every == 0
            record_due = log_due or val_due or test_due or epoch_end
            if not record_due:
                continue

            train_window.reduce()
            train_metrics = train_window.compute()
            train_window = BinaryAccumulator(runtime.device)

            val_metrics = _evaluate(model, loaders.val_loader, criterion, runtime, config) if val_due else None
            test_metrics = _evaluate(model, loaders.test_loader, criterion, runtime, config) if test_due else None
            model.train()

            row = _empty_history_row(
                global_step, epoch, float(optimizer.param_groups[0]["lr"])
            )
            _put_metrics(row, "train", train_metrics)
            if val_metrics is not None:
                _put_metrics(row, "val", val_metrics)
            if test_metrics is not None:
                _put_metrics(row, "test", test_metrics)
            history.append(row)
            if val_metrics is not None:
                metric_details.append(
                    {"step": global_step, "epoch": epoch, "split": "val", "metrics": val_metrics}
                )
            if test_metrics is not None:
                metric_details.append(
                    {"step": global_step, "epoch": epoch, "split": "test", "metrics": test_metrics}
                )

            if runtime.is_main:
                print(_format_metrics(row), flush=True)
                write_history(history, run_dir)
                write_metric_details(metric_details, run_dir)

                if val_metrics is not None:
                    better = row["val_f1"] > best_val_f1 or (
                        row["val_f1"] == best_val_f1 and row["val_loss"] < best_val_loss
                    )
                    if better:
                        best_val_f1 = float(row["val_f1"])
                        best_val_loss = float(row["val_loss"])
                        best_val_step = global_step
                        best_val_metrics = val_metrics
                        save_model_weights(model, run_dir / "checkpoints" / "best_model.pt")

                # Mid-epoch validation is a useful recovery point. Epoch-end state is
                # saved below, after scheduler.step(), so the checkpoint is internally
                # consistent with the next epoch's learning-rate state.
                if val_due and not epoch_end:
                    save_model_weights(model, run_dir / "checkpoints" / "last_model.pt")
                    save_training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        history=history,
                        config=config,
                        path=run_dir / "checkpoints" / "last_checkpoint.pt",
                    )

        if scheduler is not None:
            scheduler.step()
        if runtime.is_main:
            save_model_weights(model, run_dir / "checkpoints" / "last_model.pt")
            save_training_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                history=history,
                config=config,
                path=run_dir / "checkpoints" / "last_checkpoint.pt",
            )

    barrier()
    if runtime.is_main:
        plot_history(history, run_dir)
        last_val = next((row for row in reversed(history) if row["val_loss"] is not None), None)
        last_test = next((row for row in reversed(history) if row["test_loss"] is not None), None)
        final_val_metrics = next(
            (entry["metrics"] for entry in reversed(metric_details) if entry["split"] == "val"), None
        )
        final_test_metrics = next(
            (entry["metrics"] for entry in reversed(metric_details) if entry["split"] == "test"), None
        )
        if best_val_metrics is not None and best_val_metrics.get("per_game"):
            plot_diagnostics(
                best_val_metrics,
                run_dir / "best_val_diagnostics.png",
                title=f"Best validation diagnostics @ step {best_val_step}",
            )
        if final_test_metrics is not None and final_test_metrics.get("per_game"):
            plot_diagnostics(
                final_test_metrics,
                run_dir / "final_test_diagnostics.png",
                title="Final test diagnostics",
            )
        summary = {
            "run_dir": str(run_dir),
            "epochs": epochs,
            "global_steps": global_step,
            "duration_seconds": round(time.time() - started, 3),
            "best_val_step": best_val_step,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "final_val": last_val,
            "final_test": last_test,
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("best_model:", run_dir / "checkpoints" / "best_model.pt")
        print("loss_curve:", run_dir / "loss_curve.png")
        print("f1_curve  :", run_dir / "f1_curve.png")
        print("class_metrics_curve:", run_dir / "class_metrics_curve.png")
        print("metrics_detail:", run_dir / "metrics_detail.json")
        print("best_val_diagnostics:", run_dir / "best_val_diagnostics.png")
        print("final_test_diagnostics:", run_dir / "final_test_diagnostics.png")
        return summary
    return {"run_dir": str(run_dir), "global_steps": global_step}


@torch.no_grad()
def evaluate_checkpoint(
    config: dict[str, Any],
    runtime: Runtime,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    loaders = build_dataloaders(config, runtime)
    if split == "val":
        dataset, loader = loaders.val_dataset, loaders.val_loader
    elif split == "test":
        dataset, loader = loaders.test_dataset, loaders.test_loader
    else:
        raise ValueError("split must be val or test")

    model = build_model(config["model"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(runtime.device)
    model = wrap_ddp(
        model,
        runtime,
        find_unused_parameters=bool(config["runtime"].get("find_unused_parameters", True)),
    )
    metrics = _evaluate(model, loader, _build_criterion(config, runtime.device), runtime, config)
    if runtime.is_main:
        print(f"{split}_dataset:", json.dumps(dataset.summary(), ensure_ascii=False))
        print("metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def check_setup(config: dict[str, Any], runtime: Runtime) -> dict[str, Any]:
    loaders = build_dataloaders(config, runtime)
    overlap = loaders.train_dataset.video_keys & loaders.val_dataset.video_keys
    if overlap:
        raise RuntimeError(f"train/val video leakage detected: {sorted(overlap)[:3]}")

    model = build_model(config["model"])
    load_initial_weights(model, config["model"].get("init_weights"))
    model.to(runtime.device)
    batch = next(iter(loaders.train_loader))
    images, labels = _prepare_batch(batch, runtime.device)
    with torch.no_grad():
        logits, _ = _forward_loss(
            model,
            images,
            labels,
            _build_criterion(config, runtime.device),
            runtime,
            config["runtime"],
        )
    result = {
        "device": str(runtime.device),
        "world_size": runtime.world_size,
        "backend": runtime.backend,
        "train": loaders.train_dataset.summary(),
        "val": loaders.val_dataset.summary(),
        "test": loaders.test_dataset.summary(),
        "train_val_video_overlap": 0,
        "batch_shape": list(images.shape),
        "logits_shape": list(logits.shape),
    }
    if runtime.is_main:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
