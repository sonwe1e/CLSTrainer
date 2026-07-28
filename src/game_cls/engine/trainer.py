from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any

from game_cls.data.collate import pair_collate
from game_cls.data.pair_dataset import PairDataset, enumerate_pairs
from game_cls.data.pair_sampler import BalancedDistributedPairBatchSampler
from game_cls.engine.checkpoint import (
    restore_training_checkpoint,
    save_checkpoint_pair,
    unwrap_model,
)
from game_cls.engine.device import autocast_context, initialize_device
from game_cls.engine.distributed import (
    cleanup_distributed,
    distributed_barrier,
    distributed_context,
)
from game_cls.engine.evaluator import evaluate
from game_cls.losses.threshold_loss import combined_loss
from game_cls.model.builder import build_model
from game_cls.model.checkpoint_loader import load_model_checkpoint
from game_cls.model.freeze_policy import (
    assert_frozen_parameters_unchanged,
    configure_trainable_parameters,
    set_frozen_backbone_train_mode,
    snapshot_frozen_parameters,
)
from game_cls.reports.error_writer import write_evaluation_report


class SyntheticPairDataset:
    def __init__(self, length: int, height: int, width: int, seed: int) -> None:
        self.length = length
        self.height = height
        self.width = width
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
            (2, 3, self.height, self.width),
            dtype=torch.uint8,
            generator=generator,
        )
        if label:
            images[:, 0, : self.height // 2] += 128
        return {
            "images": images,
            "label": label,
            "meta": {
                "game": f"synthetic_{index % 2}",
                "video_id": f"{index % 4:02d}",
                "frame0_id": index,
                "frame1_id": index + 2,
                "delta": 2,
            },
        }


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataloaders(config: dict, rank: int, world_size: int):
    import torch
    from torch.utils.data import DataLoader

    data_cfg = config["data"]
    train_cfg = config["train"]
    loader_cfg = config["dataloader"]
    batch_size = int(train_cfg["local_batch_size"])
    workers = int(loader_cfg.get("num_workers", 0))
    common = {
        "num_workers": workers,
        "pin_memory": loader_cfg.get("pin_memory", False),
        "collate_fn": pair_collate,
    }
    if workers > 0:
        common["persistent_workers"] = loader_cfg.get("persistent_workers", True)
        common["prefetch_factor"] = loader_cfg.get("prefetch_factor", 2)

    if data_cfg.get("synthetic", False):
        train_length = max(
            batch_size * int(train_cfg["steps_per_epoch"]) * world_size, 128
        )
        train_dataset = SyntheticPairDataset(
            train_length, data_cfg["height"], data_cfg["width"], config["experiment"]["seed"]
        )
        test_dataset = SyntheticPairDataset(
            64, data_cfg["height"], data_cfg["width"], config["experiment"]["seed"] + 99
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **common,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, **common)
        return train_loader, test_loader, None

    from game_cls.data.augment import ConsistentPairAugment
    from game_cls.data.indexing import read_frame_parquet

    train_frames = read_frame_parquet(data_cfg["train_index"])
    test_frames = read_frame_parquet(data_cfg["test_index"])
    delta_probability = {
        int(key): float(value)
        for key, value in config["pair"]["train_delta_probability"].items()
    }
    train_pairs = enumerate_pairs(train_frames, list(delta_probability))
    test_pairs = enumerate_pairs(test_frames, [int(config["pair"]["test_delta"])])
    if not train_pairs or not test_pairs:
        raise RuntimeError("Train and test indexes must both contain legal pairs")
    transform = None
    if config.get("augmentation", {}).get("enabled", True):
        transform = ConsistentPairAugment(config["augmentation"])
    train_dataset = PairDataset(train_pairs, transform=transform)
    test_dataset = PairDataset(test_pairs)
    sampler_cfg = config["sampler"]
    sampler = BalancedDistributedPairBatchSampler(
        train_pairs,
        local_batch_size=batch_size,
        steps_per_epoch=int(train_cfg["steps_per_epoch"]),
        rank=rank,
        world_size=world_size,
        seed=config["experiment"]["seed"],
        game_alpha=sampler_cfg.get("game_alpha", 0.25),
        class_probability={
            int(key): float(value)
            for key, value in sampler_cfg["class_probability"].items()
        },
        delta_probability=delta_probability,
        deduplicate_within_global_batch=sampler_cfg.get(
            "deduplicate_within_global_batch", True
        ),
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **common)
    return train_loader, test_loader, sampler


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


def run_training(config: dict[str, Any]) -> dict:
    import torch

    rank, world_size, local_rank = distributed_context(config.get("distributed", {}))
    try:
        seed = int(config["experiment"]["seed"])
        _seed_everything(seed)
        device = initialize_device(config["device"]["accelerator"], local_rank)
        output_dir = Path(config["experiment"]["output_dir"])
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        model = build_model(config["model"])
        checkpoint_path = config["model"].get("checkpoint_path")
        if checkpoint_path:
            report = load_model_checkpoint(model, checkpoint_path)
            if rank == 0:
                print(f"Loaded {len(report.loaded)} model tensors")
                print(f"Missing: {report.missing}")
                print(f"Unexpected: {report.unexpected}")
                print(f"Shape mismatch: {report.shape_mismatch}")
        summary = configure_trainable_parameters(
            model, config["model"].get("trainable_name_contains", "cls")
        )
        set_frozen_backbone_train_mode(
            model,
            config["model"].get("trainable_name_contains", "cls"),
            config["model"].get("freeze_batchnorm_stats", True),
        )
        model.to(device)
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel

            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        frozen_snapshot = (
            snapshot_frozen_parameters(model)
            if config["train"].get("verify_frozen_parameters", False)
            else None
        )
        if rank == 0:
            print("Trainable parameters:")
            for name in summary.trainable_names:
                print(f"  {name}")
            print(
                f"Trainable={summary.trainable_count:,} Frozen={summary.frozen_count:,} "
                f"Ratio={summary.trainable_ratio:.4%}"
            )

        train_loader, test_loader, sampler = _make_dataloaders(config, rank, world_size)
        train_cfg = config["train"]
        total_steps = int(
            train_cfg.get("max_steps")
            or int(train_cfg["epochs"]) * int(train_cfg["steps_per_epoch"])
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        scheduler = _build_scheduler(optimizer, config["scheduler"], total_steps)
        use_amp = bool(config["device"].get("amp", False))
        scaler = torch.amp.GradScaler(
            device.type, enabled=use_amp and config["device"]["amp_dtype"] == "float16"
        )
        global_step = 0
        best_f1 = -1.0
        best_metrics: dict = {}
        epoch = 0
        resume_path = train_cfg.get("resume_path")
        if resume_path:
            checkpoint = restore_training_checkpoint(
                resume_path, model, optimizer, scheduler, scaler
            )
            global_step = int(checkpoint.get("global_step", 0))
            epoch = int(checkpoint.get("sampler_epoch", checkpoint.get("epoch", 0)))
            best_metrics = dict(checkpoint.get("best_metrics", {}))
            best_f1 = float(best_metrics.get("f1", -1.0))
            random_state = checkpoint.get("random_state", {})
            if random_state.get("python") is not None:
                random.setstate(random_state["python"])
            if random_state.get("numpy") is not None:
                import numpy as np

                np.random.set_state(random_state["numpy"])
            if random_state.get("torch") is not None:
                torch.set_rng_state(random_state["torch"])
            if rank == 0:
                print(f"Resumed from {resume_path} at step={global_step}, epoch={epoch}")
        if global_step >= total_steps:
            raise ValueError(
                f"Resume step {global_step} is not below target max step {total_steps}"
            )
        started = time.perf_counter()
        while global_step < total_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            set_frozen_backbone_train_mode(
                model,
                config["model"].get("trainable_name_contains", "cls"),
                config["model"].get("freeze_batchnorm_stats", True),
            )
            for batch in train_loader:
                images = batch["images"].to(device, non_blocking=True)
                images = images.float().div_(255.0) if images.dtype == torch.uint8 else images
                labels = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
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
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg.get("gradient_clip_norm", 5.0),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                if rank == 0 and global_step % train_cfg["log_every_steps"] == 0:
                    elapsed = time.perf_counter() - started
                    samples = global_step * train_cfg["local_batch_size"] * world_size
                    print(
                        f"step={global_step}/{total_steps} loss={loss.detach().item():.6f} "
                        f"ce={components['cross_entropy'].item():.6f} "
                        f"threshold_weight={components['threshold_weight']:.4f} "
                        f"samples/s={samples / elapsed:.2f}",
                        flush=True,
                    )
                eval_every = config["evaluation"].get("quick_test_every_steps", 0)
                if eval_every and global_step % eval_every == 0:
                    distributed_barrier()
                    if rank == 0:
                        metrics, errors = evaluate(
                            unwrap_model(model),
                            test_loader,
                            device,
                            config["evaluation"].get("threshold", 0.99),
                        )
                        report_dir = output_dir / "reports" / f"test_step_{global_step:08d}"
                        write_evaluation_report(report_dir, metrics, errors)
                        if metrics["f1"] > best_f1:
                            best_f1, best_metrics = metrics["f1"], metrics
                            if config["checkpoint"].get("save_best_test_f1", True):
                                save_checkpoint_pair(
                                    output_dir / "checkpoints",
                                    "best_f1_tau099",
                                    model,
                                    optimizer,
                                    scheduler,
                                    scaler,
                                    epoch,
                                    global_step,
                                    best_metrics,
                                    config,
                                )
                        set_frozen_backbone_train_mode(
                            model,
                            config["model"].get("trainable_name_contains", "cls"),
                            config["model"].get("freeze_batchnorm_stats", True),
                        )
                    distributed_barrier()
                save_every = config["checkpoint"].get("save_last_every_steps", 0)
                if rank == 0 and save_every and global_step % save_every == 0:
                    save_checkpoint_pair(
                        output_dir / "checkpoints",
                        "last",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_metrics,
                        config,
                    )
                if global_step >= total_steps:
                    break
            epoch += 1

        distributed_barrier()
        if rank == 0:
            if config["evaluation"].get("full_test_at_end", True):
                metrics, errors = evaluate(
                    unwrap_model(model),
                    test_loader,
                    device,
                    config["evaluation"].get("threshold", 0.99),
                )
                write_evaluation_report(output_dir / "reports" / "test_final", metrics, errors)
            save_checkpoint_pair(
                output_dir / "checkpoints",
                "last",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_metrics,
                config,
            )
            if frozen_snapshot is not None:
                assert_frozen_parameters_unchanged(frozen_snapshot, model)
                print("Verified: every frozen parameter remained bitwise unchanged.")
        distributed_barrier()
        return {"global_step": global_step, "best_metrics": best_metrics}
    finally:
        cleanup_distributed()
