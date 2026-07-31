from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from game_cls.contracts.task import StepContext
from game_cls.data.collate import pair_collate
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.checkpoint import (
    capture_random_state,
    clone_checkpoint_pair,
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
    train: Any
    quick_test: Any
    full_test: Any
    sampler: Any
    data_summary: dict


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


def _append_training_metrics(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        stream.write("\n")


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
        "selection_metric", "global_f1_tau099"
    )
    supported_selection_metrics = {
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
            for split in ("train", "test"):
                packed_index = Path(data_cfg[f"{split}_packed_index"])
                packed_video_index = data_cfg.get(
                    f"{split}_packed_video_index"
                ) or str(
                    packed_index.with_name("packed_video_entries.parquet")
                )
                if not Path(packed_video_index).is_file():
                    raise FileNotFoundError(
                        "Packed backend requires the integer video index: "
                        f"{packed_video_index}"
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


def _make_dataloaders(
    config: dict, rank: int, world_size: int
) -> LoaderBundle:
    from torch.utils.data import DataLoader, Subset

    from game_cls.data.video_sampler import DeterministicIndexBatchSampler

    data_cfg = config["data"]
    image_spec = ImageSpec.from_config(data_cfg)
    train_cfg = config["train"]
    batch_size = int(train_cfg["local_batch_size"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    train_common = _loader_common(config, role="train")
    eval_common = _loader_common(config, role="eval")

    if data_cfg.get("synthetic", False):
        train_length = max(batch_size * steps_per_epoch * world_size, 128)
        train_dataset = SyntheticPairDataset(
            train_length,
            image_spec,
            config["experiment"]["seed"],
        )
        test_dataset = SyntheticPairDataset(
            64,
            image_spec,
            config["experiment"]["seed"] + 99,
        )
        sampler = DeterministicIndexBatchSampler(
            train_length,
            batch_size,
            steps_per_epoch,
            rank=rank,
            world_size=world_size,
            seed=config["experiment"]["seed"],
        )
        full_indices = list(range(rank, len(test_dataset), world_size))
        quick_global = min(
            len(test_dataset),
            int(config["evaluation"].get("quick_test_pairs_per_video", 16)) * 2,
        )
        quick_indices = list(range(rank, quick_global, world_size))
        return LoaderBundle(
            train=DataLoader(
                train_dataset, batch_sampler=sampler, **train_common
            ),
            quick_test=DataLoader(
                Subset(test_dataset, quick_indices),
                batch_size=batch_size,
                **eval_common,
            ),
            full_test=DataLoader(
                Subset(test_dataset, full_indices),
                batch_size=batch_size,
                **eval_common,
            ),
            sampler=sampler,
            data_summary={
                "storage": "synthetic",
                "train_samples": train_length,
                "quick_test_samples_global": quick_global,
                "full_test_samples_global": len(test_dataset),
            },
        )

    from game_cls.data.augment import ConsistentPairAugment
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import (
        audit_warning_messages,
        validate_audit_file,
    )
    from game_cls.data.lazy_pair_dataset import (
        LazyTrainingPairDataset,
        build_eval_dataset,
    )
    from game_cls.data.video_index import (
        read_video_entries_parquet,
        video_index_memory_bytes,
    )
    from game_cls.data.video_sampler import VideoBalancedPairBatchSampler

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
    backend_name = data_cfg.get("backend", "png")
    if backend_name == "packed_uint8":
        train_video_index = data_cfg.get(
            "train_packed_video_index"
        ) or str(
            Path(data_cfg["train_packed_index"]).with_name(
                "packed_video_entries.parquet"
            )
        )
        test_video_index = data_cfg.get("test_packed_video_index") or str(
            Path(data_cfg["test_packed_index"]).with_name(
                "packed_video_entries.parquet"
            )
        )
    else:
        train_video_index = data_cfg.get("train_video_index")
        test_video_index = data_cfg.get("test_video_index")
    train_videos = read_video_entries_parquet(
        train_video_index
        if train_video_index and Path(train_video_index).is_file()
        else data_cfg["train_index"],
        delta_probability.keys(),
    )
    test_delta = int(config["pair"]["test_delta"])
    test_videos = read_video_entries_parquet(
        test_video_index
        if test_video_index and Path(test_video_index).is_file()
        else data_cfg["test_index"],
        (test_delta,),
    )
    transform = None
    if config.get("augmentation", {}).get("enabled", True):
        transform = ConsistentPairAugment(config["augmentation"])
    if backend_name == "png":
        train_decoder = None
        test_decoder = None
    elif backend_name == "packed_uint8":
        from game_cls.data.packed_backend import PackedUint8Backend

        train_decoder = PackedUint8Backend(
            data_cfg["train_packed_index"],
            image_spec=image_spec,
            max_open_shards=int(
                data_cfg.get("packed_max_open_shards", 16)
            ),
        )
        test_decoder = PackedUint8Backend(
            data_cfg["test_packed_index"],
            image_spec=image_spec,
            max_open_shards=int(
                data_cfg.get("packed_max_open_shards", 16)
            ),
        )
    else:
        raise ValueError(f"Unsupported data backend: {backend_name}")
    train_dataset = LazyTrainingPairDataset(
        train_videos, transform=transform, decoder=train_decoder
    )
    sampler_cfg = config["sampler"]
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
        deduplicate_within_global_batch=sampler_cfg.get(
            "deduplicate_within_global_batch", True
        ),
    )
    quick_dataset = build_eval_dataset(
        test_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=int(
            config["evaluation"].get("quick_test_pairs_per_video", 128)
        ),
        decoder=test_decoder,
    )
    full_dataset = build_eval_dataset(
        test_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        decoder=test_decoder,
    )
    global_quick = _distributed_sum_int(len(quick_dataset))
    global_full = _distributed_sum_int(len(full_dataset))
    if global_quick <= 0 or global_full <= 0:
        raise RuntimeError("Test index does not contain legal delta=2 pairs")
    return LoaderBundle(
        train=DataLoader(
            train_dataset, batch_sampler=sampler, **train_common
        ),
        quick_test=DataLoader(
            quick_dataset, batch_size=batch_size, **eval_common
        ),
        full_test=DataLoader(
            full_dataset, batch_size=batch_size, **eval_common
        ),
        sampler=sampler,
        data_summary={
            "storage": "video_index_lazy_pairs",
            "image_backend": backend_name,
            "train_videos": len(train_videos),
            "test_videos": len(test_videos),
            "video_index_payload_bytes_per_rank_estimate": (
                video_index_memory_bytes(train_videos)
                + video_index_memory_bytes(test_videos)
            ),
            "quick_test_samples_global": global_quick,
            "full_test_samples_global": global_full,
            "full_test_index_bytes_this_rank": full_dataset.index_nbytes,
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


def _gather_random_states_via_runtime(runtime: Any) -> list[dict] | None:
    local = capture_random_state()
    dist = runtime.distributed
    if dist.world_size <= 1:
        return [local]
    gathered = [None for _ in range(dist.world_size)] if dist.rank == 0 else None
    dist.gather_object(local, dst=0)
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
    runtime: Any = None,
) -> None:
    if runtime is None:
        states = _gather_random_states(rank, world_size)
    else:
        states = _gather_random_states_via_runtime(runtime)
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
        "selection_metric", "global_f1_tau099"
    )
    minimum_worst = evaluation_config.get("minimum_worst_game_f1")
    eligible = (
        minimum_worst is None
        or float(metrics.get("worst_game_f1_tau099", 0.0))
        >= float(minimum_worst)
    )
    if metric_name == "composite":
        weights = evaluation_config.get("selection_weights", {})
        components = {
            "global_f1": float(metrics.get("global_f1_tau099", 0.0)),
            "macro_game_f1": float(
                metrics.get("macro_game_f1_tau099", 0.0)
            ),
            "worst_game_f1": float(
                metrics.get("worst_game_f1_tau099", 0.0)
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
        if metric_name not in metrics:
            raise KeyError(
                f"Selection metric is absent from evaluation output: {metric_name}"
            )
        score = float(metrics[metric_name])
    return score, eligible


def _annotate_selection(metrics: dict, evaluation_config: dict) -> dict:
    score, eligible = _selection_score(metrics, evaluation_config)
    metrics["selection_metric"] = evaluation_config.get(
        "selection_metric", "global_f1_tau099"
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
) -> EvaluationOutput:
    report_dir = output_dir / "reports" / f"{kind}_step_{global_step:08d}"
    prepare_evaluation_directory(report_dir, rank)
    distributed_barrier()
    result = evaluate(
        unwrap_model(model),
        dataloader,
        device,
        config["evaluation"].get("threshold", 0.99),
        checkpoint_step=global_step,
        distributed=is_distributed(),
        rank=rank,
        world_size=world_size,
        evaluation_kind=kind,
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
    )
    distributed_barrier()
    if rank == 0:
        metrics = dict(result.metrics or {})
        metrics.update(
            {
                "evaluation_kind": kind,
                "evaluation_role": "observed_dev_test",
                "checkpoint_step": global_step,
                "evaluation_amp": bool(
                    config["evaluation"].get(
                        "amp", config["device"].get("amp", False)
                    )
                ),
                "evaluation_amp_dtype": str(
                    config["evaluation"].get(
                        "amp_dtype",
                        config["device"].get(
                            "amp_dtype", "bfloat16"
                        ),
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
            lightweight=kind == "quick",
            html_max_errors=int(
                config["evaluation"].get("html_max_errors_per_group", 200)
            ),
            preview_decoder=getattr(
                getattr(dataloader, "dataset", None), "decoder", None
            ),
        )
        result.metrics = metrics
    distributed_barrier()
    return result


def run_training(config: dict[str, Any]) -> dict:
    """Compatibility entry point (USERPLAN §9.4).

    Builds the extensible component graph and delegates to an
    :class:`ExperimentRunner`. The external signature is unchanged so
    ``tools/train.py`` and existing callers keep working.
    """
    from .runner import ExperimentRunner

    components = _build_components_for_runner(config)
    runner = ExperimentRunner(components)
    try:
        runner.setup()
        return runner.run()
    finally:
        runner.close()


def _run_training_loop_legacy(config: dict[str, Any]) -> dict:
    """Standalone entry point that runs the loop without a component runtime.

    Used by callers that go through ``_run_training_loop`` directly without
    building the full component graph. Preserves the original behavior.
    """
    return _run_training_loop(config)


def _build_components_for_runner(config: dict[str, Any]) -> Any:
    """Build the task and runtime components the runner needs."""
    from game_cls.data.image_spec import ImageSpec

    from .builders import ExperimentComponents
    from .runner import ExperimentRunner  # noqa: F401

    image_spec = ImageSpec.from_config(config["data"])
    from game_cls.tasks.dual_frame_binary import DualFrameBinaryTask

    task = DualFrameBinaryTask(image_spec=image_spec, loss_config=config["loss"])
    runtime = _build_runtime(config)

    class _Cfg:
        pass

    cfg = _Cfg()
    return ExperimentComponents(
        config=cfg,
        raw_config=config,
        runtime=runtime,
        task=task,
        trainable_policy=_build_trainable_policy(config),
        model=None,
        image_spec=image_spec,
    )


def _build_runtime(config: dict[str, Any]) -> Any:
    from game_cls.runtime.factories import build_runtime

    runtime_cfg = config.get("runtime", {})
    accelerator = runtime_cfg.get("accelerator", {}).get(
        "type", config.get("device", {}).get("accelerator", "cpu")
    )
    distributed_cfg = runtime_cfg.get("distributed", {})
    distributed = distributed_cfg.get("type", "single_process")
    distributed_params = dict(distributed_cfg.get("params", {}) or {})

    class _Sel:
        pass

    acc_sel = _Sel()
    acc_sel.type = accelerator
    dist_sel = _Sel()
    dist_sel.type = distributed
    dist_sel.params = distributed_params
    runtime_sel = _Sel()
    runtime_sel.accelerator = acc_sel
    runtime_sel.distributed = dist_sel
    return build_runtime(runtime_sel)


def _build_trainable_policy(config: dict[str, Any]) -> Any:
    from game_cls.trainable.build import build_trainable_policy

    trainable_cfg = config.get("trainable", {})
    policy = trainable_cfg.get("policy", {})

    class _Sel:
        pass

    sel = _Sel()
    sel.type = policy.get("type", "name_token")
    sel.factory = policy.get("factory", "")
    sel.params = dict(policy.get("params", {}) or {})
    return build_trainable_policy(sel)


def _run_training_loop(
    config: dict[str, Any], *, runtime: Any = None
) -> dict:
    import torch

    validate_training_config(config)
    image_spec = ImageSpec.from_config(config["data"])
    from game_cls.tasks.dual_frame_binary import DualFrameBinaryTask

    task = DualFrameBinaryTask(
        image_spec=image_spec,
        loss_config=config["loss"],
        task_config=None,
    )

    # When a runtime is supplied (from the runner's component graph) we use it
    # directly; otherwise we fall back to the legacy ``initialize_runtime``
    # path so the loop still works as a standalone entry point.
    if runtime is None:
        rank, world_size, local_rank, device = initialize_runtime(config)
    else:
        rank = int(runtime.distributed.rank)
        world_size = int(runtime.distributed.world_size)
        local_rank = int(runtime.distributed.local_rank)
        device = runtime.accelerator.device

    try:
        seed = int(config["experiment"]["seed"])
        _seed_everything(seed + rank)
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
        if runtime is None:
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
        else:
            model = runtime.wrap_model(model)
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
        if runtime is None:
            scaler = torch.amp.GradScaler(
                device.type, enabled=use_amp and config["device"]["amp_dtype"] == "float16"
            )
        else:
            scaler = runtime.accelerator.make_grad_scaler(use_amp, config["device"].get("amp_dtype", "float16"))
        global_step = 0
        epoch = 0
        step_in_epoch = 0
        best_metrics: dict = {}
        evaluation_state = {
            "quick_test_count": 0,
            "full_test_count": 0,
            "last_full_metrics": {},
            "best_observed_dev_test_metrics": {},
            "train_delta_history": [],
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
        while global_step < run_until_step:
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
                if not input_shape_validated:
                    task.validate_cpu_batch(batch)
                    input_shape_validated = True
                context = StepContext(
                    global_step=global_step,
                    total_steps=total_steps,
                    epoch=epoch,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=config["device"].get("amp_dtype", "float16"),
                )
                device_batch = task.move_batch_to_device(batch, context)
                timing["host_h2d_enqueue"] += (
                    time.perf_counter() - transfer_started
                )
                optimizer.zero_grad(set_to_none=True)
                forward_started = time.perf_counter()
                autocast_ctx = (
                    runtime.autocast(use_amp, config["device"].get("amp_dtype", "float16"))
                    if runtime is not None
                    else autocast_context(
                        device, use_amp, config["device"].get("amp_dtype", "float16")
                    )
                )
                with autocast_ctx:
                    task_output = task.forward(model, device_batch, context)
                    loss_output = task.compute_loss(task_output, device_batch, context)
                timing["host_forward_enqueue"] += (
                    time.perf_counter() - forward_started
                )
                backward_started = time.perf_counter()
                scaler.scale(loss_output.total).backward()
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

                log_every = int(train_cfg["log_every_steps"])

                quick_every = int(
                    config["evaluation"].get("quick_test_every_steps", 0)
                )
                full_every = int(
                    config["evaluation"].get("full_test_every_steps", 0)
                )
                run_full = bool(
                    full_every and global_step % full_every == 0
                )
                run_quick = bool(
                    quick_every
                    and global_step % quick_every == 0
                    and not run_full
                )
                if run_quick:
                    evaluation_started = time.perf_counter()
                    result = _run_evaluation(
                        kind="quick",
                        model=model,
                        dataloader=loaders.quick_test,
                        device=device,
                        config=config,
                        output_dir=output_dir,
                        global_step=global_step,
                        rank=rank,
                        world_size=world_size,
                    )
                    evaluation_state["quick_test_count"] += 1
                    _set_train_mode(model, config["model"])
                    evaluation_seconds += (
                        time.perf_counter() - evaluation_started
                    )

                is_best = False
                if run_full:
                    evaluation_started = time.perf_counter()
                    result = _run_evaluation(
                        kind="full",
                        model=model,
                        dataloader=loaders.full_test,
                        device=device,
                        config=config,
                        output_dir=output_dir,
                        global_step=global_step,
                        rank=rank,
                        world_size=world_size,
                    )
                    evaluation_state["full_test_count"] += 1
                    if runtime is None:
                        metrics = _broadcast_object(result.metrics, rank)
                    else:
                        payload = [result.metrics if rank == 0 else None]
                        runtime.distributed.broadcast_object_list(payload, src=0)
                        metrics = payload[0]
                    evaluation_state["last_full_metrics"] = metrics
                    is_best = _is_better_model(
                        metrics, best_metrics, config["evaluation"]
                    )
                    if is_best:
                        best_metrics = metrics
                        evaluation_state["best_observed_dev_test_metrics"] = metrics
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
                checkpoint_started = None
                if is_best and _save_best_enabled(config["checkpoint"]):
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
                        runtime=runtime,
                    )
                    if rank == 0:
                        clone_checkpoint_pair(
                            output_dir / "checkpoints",
                            "last",
                            "best_observed_dev_test_selection",
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
                        runtime=runtime,
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
                    if runtime is None:
                        _synchronize_device_for_metrics(device)
                    else:
                        runtime.synchronize()
                    if rank == 0:
                        now = time.perf_counter()
                        interval_seconds = now - log_interval_start
                        interval_steps = max(1, timing_steps)
                        elapsed = now - started
                        averages = {
                            key: value / interval_steps
                            for key, value in timing.items()
                        }
                        loss_value = float(loss_output.total.detach().item())
                        ce_value = float(
                            loss_output.components["cross_entropy"].item()
                        )
                        threshold_loss_value = float(
                            loss_output.components["threshold_loss"].item()
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
                            "loss": loss_value,
                            "ce": ce_value,
                            "threshold_loss": threshold_loss_value,
                            "threshold_weight": float(
                                loss_output.components["threshold_weight"]
                            ),
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
                        print(
                            f"step={global_step}/{total_steps} "
                            f"loss={loss_value:.6f} "
                            f"ce={ce_value:.6f} "
                            f"threshold_loss={threshold_loss_value:.6f} "
                            f"threshold_weight="
                            f"{loss_output.components['threshold_weight']:.4f} "
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
                    log_interval_start = time.perf_counter()
                    log_interval_samples = 0
                    evaluation_seconds = 0.0
                    checkpoint_seconds = 0.0
                    timing = {key: 0.0 for key in timing}
                    timing_steps = 0
                last_batch_finished = time.perf_counter()
                if global_step >= run_until_step:
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
                    if rank == 0:
                        print(
                            "Observed train delta distribution:",
                            json.dumps(distribution, ensure_ascii=False),
                        )
                epoch += 1
                step_in_epoch = 0

        final_is_best = False
        if config["evaluation"].get("full_test_at_end", True):
            already_full = (
                evaluation_state["last_full_metrics"].get("checkpoint_step")
                == global_step
            )
            if not already_full:
                result = _run_evaluation(
                    kind="full_final",
                    model=model,
                    dataloader=loaders.full_test,
                    device=device,
                    config=config,
                    output_dir=output_dir,
                    global_step=global_step,
                    rank=rank,
                    world_size=world_size,
                )
                evaluation_state["full_test_count"] += 1
                if runtime is None:
                    final_metrics = _broadcast_object(result.metrics, rank)
                else:
                    payload = [result.metrics if rank == 0 else None]
                    runtime.distributed.broadcast_object_list(payload, src=0)
                    final_metrics = payload[0]
                evaluation_state["last_full_metrics"] = final_metrics
                if _is_better_model(
                    final_metrics, best_metrics, config["evaluation"]
                ):
                    best_metrics = final_metrics
                    evaluation_state["best_observed_dev_test_metrics"] = final_metrics
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
            runtime=runtime,
        )
        if (
            final_is_best
            and _save_best_enabled(config["checkpoint"])
            and rank == 0
        ):
            clone_checkpoint_pair(
                output_dir / "checkpoints",
                "last",
                "best_observed_dev_test_selection",
            )
        if runtime is None:
            distributed_barrier()
        else:
            runtime.barrier()
        if rank == 0:
            if frozen_snapshot is not None:
                assert_frozen_parameters_unchanged(frozen_snapshot, model)
                print("Verified: every frozen parameter remained bitwise unchanged.")
            summary_payload = {
                "global_step": global_step,
                "last_checkpoint_metrics": evaluation_state["last_full_metrics"],
                "best_observed_dev_test_metrics": best_metrics,
                "test_evaluation_counts": {
                    "quick": evaluation_state["quick_test_count"],
                    "full": evaluation_state["full_test_count"],
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
        if runtime is None:
            distributed_barrier()
        else:
            runtime.barrier()
        return {
            "global_step": global_step,
            "last_metrics": evaluation_state["last_full_metrics"],
            "best_metrics": best_metrics,
            "evaluation_state": evaluation_state,
        }
    finally:
        if runtime is None:
            cleanup_distributed()
        else:
            runtime.cleanup()
