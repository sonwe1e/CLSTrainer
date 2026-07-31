"""Legacy data loading pipeline extracted from ``engine.trainer``.

This module holds the original ``_make_dataloaders`` logic so that both the
training loop (``engine.trainer``) and the default ``DataModule``
(``LegacyGameVideoDataModule``) can share it without a circular import::

    Trainer ──► DataModule ──► legacy_pipeline ◄── Trainer (fallback)

Previously ``LegacyGameVideoDataModule.build_loaders`` imported
``_make_dataloaders`` from ``engine.trainer``, creating a dependency cycle
(Trainer → DataModule → Trainer). Now both depend on this leaf module.
"""
from __future__ import annotations

from functools import partial
from typing import Any

from game_cls.data.collate import pair_collate
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.distributed import is_distributed


class SyntheticPairDataset:
    """In-memory synthetic dataset used by smoke tests."""

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


def _distributed_sum_int(value: int) -> int:
    if not is_distributed():
        return value
    import torch.distributed as dist

    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return sum(int(item) for item in values)


def build_legacy_loader_bundle(
    config: dict,
    image_spec: Any,
    runtime: Any,
) -> Any:
    """Build the train/quick/full dataloaders using the legacy pipeline.

    This is the single source of truth for the default data loading behavior.
    Both ``LegacyGameVideoDataModule.build_loaders`` and the standalone
    ``_make_dataloaders`` in ``engine.trainer`` delegate here.
    """
    from torch.utils.data import DataLoader, Subset

    from game_cls.contracts.data import LoaderBundle
    from game_cls.data.video_sampler import DeterministicIndexBatchSampler

    rank = int(runtime.distributed.rank)
    world_size = int(runtime.distributed.world_size)
    data_cfg = config["data"]
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
    from game_cls.data.backends.registry import build_backend_from_legacy_data_config
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
    if data_cfg.get("strict_audit", True):
        from pathlib import Path

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
        from pathlib import Path

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
    from pathlib import Path

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
    # Use BackendFactory to create the frame backend.
    backend_selector = config.get("data", {}).get("backend", {})
    if isinstance(backend_selector, dict):
        pass
    else:
        backend_selector = {"type": str(backend_selector), "params": {}}
    train_decoder = build_backend_from_legacy_data_config(
        data_cfg, backend_selector, "train", image_spec
    )
    test_decoder = build_backend_from_legacy_data_config(
        data_cfg, backend_selector, "test", image_spec
    )
    train_dataset = LazyTrainingPairDataset(
        train_videos, transform=transform, decoder=train_decoder
    )
    # Resolve sampler params from either V2 selector (sampler.policy.params)
    # or legacy flat fields (sampler.game_alpha, ...). This lets both native
    # V2 configs and migrated V1 configs drive the same pipeline.
    sampler_cfg = config["sampler"]
    policy_cfg = sampler_cfg.get("policy", {}) if isinstance(sampler_cfg, dict) else {}
    policy_params = dict(policy_cfg.get("params", {}) or {})
    # V2 selector takes precedence; legacy flat fields are the fallback.
    resolved_params = {
        key: policy_params[key] if key in policy_params else sampler_cfg.get(key)
        for key in ("game_alpha", "class_probability", "deduplicate_within_global_batch")
    }
    # Use the SamplingPolicy interface (USERPLAN §11 D3). The policy wraps
    # the sampling algorithm; PolicyBatchSampler adapts it to PyTorch's
    # BatchSampler protocol so the DataLoader sees a single object. This lets
    # new sampling algorithms be added by implementing the policy protocol —
    # the training loop and DataModule never branch on the algorithm name.
    from game_cls.data.policy_batch_sampler import PolicyBatchSampler
    from game_cls.data.sampling.balanced_game_label_delta import (
        BalancedGameLabelDeltaPolicy,
    )

    sampling_policy = BalancedGameLabelDeltaPolicy.from_config(
        train_videos,
        batch_size,
        steps_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=config["experiment"]["seed"],
        game_alpha=resolved_params.get("game_alpha", 0.25),
        class_probability={
            int(key): float(value)
            for key, value in resolved_params["class_probability"].items()
        }
        if resolved_params.get("class_probability")
        else None,
        delta_probability=delta_probability,
        deduplicate_within_global_batch=resolved_params.get(
            "deduplicate_within_global_batch", True
        ),
    )
    # Wrap the policy in PolicyBatchSampler so the DataLoader depends only on
    # the BatchSampler protocol, not on the concrete policy type. The wrapper
    # delegates to the policy's own __iter__/__len__ when available (preserving
    # exact-resume semantics).
    sampler = PolicyBatchSampler(
        policy=sampling_policy,
        catalog=None,  # BalancedGameLabelDeltaPolicy uses its own index
        steps_per_epoch=steps_per_epoch,
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
