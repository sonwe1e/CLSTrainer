from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from game_cls.config_schema import finalize_config, resolve_source_identity_namespaces
from game_cls.data.collate import pair_collate
from game_cls.data.image_spec import ImageSpec
from game_cls.engine.training.config_validation import (
    _dataloader_option,
    has_independent_test,
)
from game_cls.engine.training.loop_util import _initialize_data_worker
from game_cls.engine.training.state import _distributed_sum_int
from game_cls.engine.training.synthetic import SyntheticPairDataset


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


def _loader_common(config: dict, role: str) -> dict:
    workers = int(_dataloader_option(config, role, "num_workers", 0))
    if workers < 0:
        raise ValueError(f"dataloader.{role}.num_workers must be non-negative")
    common = {
        "num_workers": workers,
        "pin_memory": bool(_dataloader_option(config, role, "pin_memory", False)),
        "collate_fn": pair_collate,
    }
    if workers == 0:
        return common

    accelerator = str(config["device"]["accelerator"])
    context = _dataloader_option(config, role, "multiprocessing_context", None)
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
            "NPU DataLoader with num_workers > 0 must use multiprocessing_context=spawn"
        )

    timeout = float(_dataloader_option(config, role, "timeout_seconds", 180))
    if timeout <= 0:
        raise ValueError(
            f"dataloader.{role}.timeout_seconds must be positive when num_workers > 0"
        )
    prefetch_factor = int(_dataloader_option(config, role, "prefetch_factor", 2))
    if prefetch_factor <= 0:
        raise ValueError(f"dataloader.{role}.prefetch_factor must be positive")
    worker_num_threads = int(_dataloader_option(config, role, "worker_num_threads", 1))
    if worker_num_threads <= 0:
        raise ValueError(f"dataloader.{role}.worker_num_threads must be positive")

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


def _build_real_data_components(config: dict, rank: int, world_size: int) -> dict:
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
    namespaces_by_split = resolve_source_identity_namespaces(
        data_cfg.get("source_video_identity")
    )
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
                data_cfg.get("require_unique_video_keys_across_splits", False)
            ),
            minimum_pairs_per_game_label_delta={
                int(key): int(value)
                for key, value in data_cfg.get(
                    "minimum_pairs_per_game_label_delta", {}
                ).items()
            },
            identity_mode=(data_cfg.get("source_video_identity") or {}).get(
                "mode", "game_video"
            ),
            namespaces_by_split=namespaces_by_split,
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
    val_videos = read_video_entries_parquet(_video_index_path("val"), (test_delta,))
    test_videos = (
        read_video_entries_parquet(_video_index_path("test"), (test_delta,))
        if independent_test
        else None
    )

    # Join the optional per-video metadata sidecar AFTER the split is
    # derived (step5 P2): it can never leak a video across splits, is not
    # part of any dedup identity, and defaults every field to None/1.0 when
    # absent — so a config without metadata_sidecar stays byte-identical.
    metadata_sidecar_path = data_cfg.get("metadata_sidecar")
    if metadata_sidecar_path:
        from game_cls.data.sidecar import (
            apply_sidecar,
            read_metadata_sidecar,
            validate_sidecar_against_index,
        )

        sidecar = read_metadata_sidecar(metadata_sidecar_path)
        all_videos = (
            train_videos + val_videos + (test_videos if test_videos is not None else [])
        )
        validate_sidecar_against_index(sidecar, all_videos)
        train_videos = apply_sidecar(train_videos, sidecar)
        val_videos = apply_sidecar(val_videos, sidecar)
        if test_videos is not None:
            test_videos = apply_sidecar(test_videos, sidecar)

    decoders: dict[str, Any] = {"train": None, "val": None, "test": None}
    if backend_name == "packed_uint8":
        from game_cls.data.packed_backend import (
            PackedUint8Backend,
            verify_packed_provenance,
        )

        # Audit P0-6: refuse stale shards before a DataLoader is built. A
        # regenerated index/audit with forgotten repacking would otherwise make
        # the framework prove the new index while the model eats old pixels.
        provenance_audit_path = data_cfg.get("audit_path") or str(
            Path(data_cfg["train_index"]).parent / "audit.json"
        )
        for split in ("train", "val", "test"):
            if split == "test" and test_videos is None:
                continue
            verify_packed_provenance(
                Path(data_cfg[f"{split}_packed_index"]).with_name(
                    "packed_manifest.json"
                ),
                data_cfg.get(f"{split}_index"),
                audit_path=provenance_audit_path,
                split_manifest_path=(data_cfg.get("split") or {}).get("manifest"),
            )
            decoders[split] = PackedUint8Backend(
                data_cfg[f"{split}_packed_index"],
                image_spec=image_spec,
                max_open_shards=int(data_cfg.get("packed_max_open_shards", 16)),
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


def _warn_subtype_grouping_unavailable(
    rank: int,
    subtype_grouping: bool,
    max_worst_subtype_fpr: Any,
    reason: str,
) -> None:
    """Warn when subtype grouping was asked for but cannot be produced.

    Without a ``game_label_subtype`` catalog the evaluator reports
    ``worst_subtype_fpr_at_decision_threshold=None``, and constrained
    selection treats that as INELIGIBLE -- so a run configured with
    ``evaluation.max_worst_subtype_fpr`` would silently never select a best
    checkpoint. Say so at setup instead of at the end of training.
    """
    if not subtype_grouping or rank != 0:
        return
    message = (
        "[WARNING] evaluation.group_by_negative_subtype is enabled but no "
        f"subtype grouping is available: {reason}. Subtype metrics will be "
        "reported as null."
    )
    if max_worst_subtype_fpr is not None:
        message += (
            " evaluation.max_worst_subtype_fpr is also set, so constrained "
            "selection will reject EVERY checkpoint and no best checkpoint "
            "will be written."
        )
    print(message, flush=True)


def _log_hard_negative_buckets(summary: dict, rank: int) -> None:
    """Print per-bucket video and legal-pair counts at startup.

    A human must be able to read "hard bucket has 3 videos, 12 legal pairs"
    on the first lines of a run instead of discovering weeks later that the
    bucket was empty and every "hard" negative was an ordinary one.
    """
    if rank != 0 or not summary.get("enabled"):
        return
    buckets = summary.get("buckets") or {}
    parts = [
        f"{bucket}={counts['videos']} videos/{counts['legal_pairs']} legal pairs"
        for bucket, counts in buckets.items()
    ]
    print(
        "[hard-negative] buckets: "
        + ", ".join(parts)
        + f" (negative_mix={summary.get('negative_mix')}, "
        + f"min_videos_per_subtype_bucket={summary.get('min_videos_per_subtype_bucket')})",
        flush=True,
    )
    for game, rows in sorted((summary.get("by_game") or {}).items()):
        detail = ", ".join(
            f"{bucket}={counts['videos']} videos/{counts['legal_pairs']} legal pairs"
            for bucket, counts in sorted(rows.items())
        )
        print(f"[hard-negative]   game {game}: {detail}", flush=True)
    excluded = int(summary.get("excluded_videos") or 0)
    if excluded:
        print(
            f"[WARNING] [hard-negative] {excluded} negative video(s) match "
            "neither hard_subtypes nor ordinary_subtypes and are excluded "
            "from negative sampling entirely.",
            flush=True,
        )
    undersized = summary.get("undersized_cells") or []
    if undersized:
        cells = ", ".join(
            f"{cell['game']}/delta{cell['delta']}/{cell['bucket']}={cell['videos']}"
            for cell in undersized[:10]
        )
        print(
            f"[WARNING] [hard-negative] {len(undersized)} (game, delta, bucket) "
            "cell(s) hold fewer videos than "
            f"min_videos_per_subtype_bucket and will borrow from the other "
            f"bucket: {cells}{'...' if len(undersized) > 10 else ''}",
            flush=True,
        )


def _require_non_degenerate_hard_negatives(summary: dict) -> None:
    """Refuse a hard-negative run whose hard bucket is globally empty.

    The sampler keeps a per-cell fallback so a game with no hard negatives
    cannot stall training, but a *globally* empty hard bucket means the
    feature does nothing at all: the config asked for hard-negative mixing
    and would get plain negative sampling.
    """
    if not summary.get("enabled"):
        return
    hard = (summary.get("buckets") or {}).get("hard") or {}
    if int(hard.get("videos", 0)) > 0 and int(hard.get("legal_pairs", 0)) > 0:
        return
    raise RuntimeError(
        "data.hard_negative.enabled=true but the hard bucket is empty "
        f"({hard.get('videos', 0)} videos, {hard.get('legal_pairs', 0)} legal "
        "pairs): no video in the train split carries a negative_subtype from "
        "data.hard_negative.hard_subtypes. Training would silently degrade to "
        "plain negative sampling. Check data.metadata_sidecar coverage and "
        "data.hard_negative.hard_subtypes, or set "
        "data.hard_negative.enabled=false."
    )


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
    dataset: Any = build_eval_dataset(
        videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=max_pairs_per_video,
        decoder=decoder,
        group_by_negative_subtype=bool(
            config["evaluation"].get("group_by_negative_subtype", False)
        ),
    )
    loader = DataLoader(dataset, batch_size=batch_size, **eval_common)
    return loader, components


def build_external_pool_loader(
    config: dict,
    *,
    pool: str,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int | None = None,
):
    """Evaluation DataLoader over an external pool (challenge / mining).

    These pools live outside train/val/test, so they carry their own video
    index, their own optional metadata sidecar, and -- under
    ``data.backend=packed_uint8`` -- their own shard index. Building them
    inline (as the benchmark CLI used to) silently dropped all three: the PNG
    decoder was used for packed data, and the missing sidecar left
    ``negative_subtype`` unset so every subtype metric came back null.

    ``pool`` is "challenge" or "mining".
    """
    from torch.utils.data import DataLoader

    from game_cls.data.lazy_pair_dataset import build_eval_dataset
    from game_cls.data.video_index import read_video_entries_parquet

    if pool not in {"challenge", "mining"}:
        raise ValueError(f"Unsupported external pool: {pool!r}")
    config = finalize_config(config)
    data_cfg = config["data"]
    backend_name = data_cfg.get("backend", "png")
    packed = backend_name == "packed_uint8"
    if backend_name not in {"png", "packed_uint8"}:
        raise ValueError(f"Unsupported data backend: {backend_name}")

    # Spelled-out key names, not f-string joins: the CI consumer guard scans
    # the source for each schema leaf as a literal, and a key that only exists
    # as "{prefix}packed_index" would read as orphaned.
    if pool == "challenge":
        source: dict = data_cfg
        section = "data"
        keys = {
            "video": "challenge_video_index",
            "packed_video": "challenge_packed_video_index",
            "packed": "challenge_packed_index",
            "metadata": "challenge_metadata",
        }
    else:
        source = data_cfg.get("mining") or {}
        section = "data.mining"
        keys = {
            "video": "pool_video_index",
            "packed_video": "pool_packed_video_index",
            "packed": "pool_packed_index",
            "metadata": "pool_metadata",
        }
    path = {name: f"{section}.{key}" for name, key in keys.items()}

    video_index = (source.get(keys["packed_video"]) if packed else None) or source.get(
        keys["video"]
    )
    if not video_index:
        wanted = (
            f"{path['packed_video']} or {path['video']}" if packed else path["video"]
        )
        raise ValueError(f"{pool} evaluation needs {wanted}.")
    packed_index = source.get(keys["packed"]) if packed else None
    if packed and not packed_index:
        raise ValueError(
            f"data.backend=packed_uint8 requires {path['packed']}. Without it "
            "the PNG decoder would be used on packed shards."
        )

    test_delta = int(config["pair"]["test_delta"])
    videos = read_video_entries_parquet(video_index, (test_delta,))

    sidecar_path = source.get(keys["metadata"])
    if sidecar_path:
        from game_cls.data.sidecar import (
            apply_sidecar,
            read_metadata_sidecar,
            validate_sidecar_against_index,
        )

        sidecar = read_metadata_sidecar(sidecar_path)
        validate_sidecar_against_index(sidecar, videos)
        videos = apply_sidecar(videos, sidecar)

    decoder = None
    if packed:
        from game_cls.data.packed_backend import PackedUint8Backend

        assert packed_index is not None
        decoder = PackedUint8Backend(
            packed_index,
            image_spec=ImageSpec.from_config(data_cfg),
            max_open_shards=int(data_cfg.get("packed_max_open_shards", 16)),
        )

    subtype_grouping = bool(
        config["evaluation"].get("group_by_negative_subtype", False)
    )
    dataset: Any = build_eval_dataset(
        videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        decoder=decoder,
        group_by_negative_subtype=subtype_grouping,
    )
    if subtype_grouping and "game_label_subtype" not in dataset.group_catalogs:
        _warn_subtype_grouping_unavailable(
            rank,
            subtype_grouping,
            config["evaluation"].get("max_worst_subtype_fpr"),
            f"no video in the {pool} pool carries a negative_subtype label "
            f"(check {path['metadata']})",
        )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config["train"]["local_batch_size"]),
        **_loader_common(config, role="eval"),
    )
    return loader, videos


def _make_dataloaders(config: dict, rank: int, world_size: int) -> LoaderBundle:
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
    subtype_grouping = bool(evaluation_cfg.get("group_by_negative_subtype", False))
    max_worst_subtype_fpr = evaluation_cfg.get("max_worst_subtype_fpr")
    probe_pairs_per_video = int(evaluation_cfg.get("train_probe_pairs_per_video", 32))
    val_quick_pairs_per_video = int(
        evaluation_cfg.get(
            "val_quick_pairs_per_video",
            evaluation_cfg.get("quick_test_pairs_per_video", 128),
        )
    )

    if data_cfg.get("synthetic", False):
        _warn_subtype_grouping_unavailable(
            rank,
            subtype_grouping,
            max_worst_subtype_fpr,
            "data.synthetic=true datasets carry no negative-subtype metadata",
        )
        train_length = max(batch_size * steps_per_epoch * world_size, 128)
        train_dataset: Any = SyntheticPairDataset(
            train_length,
            image_spec,
            config["experiment"]["seed"],
        )
        val_dataset: Any = SyntheticPairDataset(
            64,
            image_spec,
            config["experiment"]["seed"] + 99,
        )
        test_dataset: Any = SyntheticPairDataset(
            64,
            image_spec,
            config["experiment"]["seed"] + 197,
        )
        sampler: Any = DeterministicIndexBatchSampler(
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
            train=DataLoader(train_dataset, batch_sampler=sampler, **train_common),
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
    from game_cls.data.sidecar import check_hard_negative_readiness
    from game_cls.data.video_index import video_index_memory_bytes
    from game_cls.data.video_sampler import VideoBalancedPairBatchSampler

    # Filesystem half of the hard-negative contract (the structural half is in
    # config_schema.semantic_validate, which must stay filesystem-free). Run it
    # before reading any parquet so a missing sidecar names itself instead of
    # surfacing later as an empty bucket.
    readiness = check_hard_negative_readiness(config)
    if readiness:
        raise RuntimeError(
            "Hard-negative sampling is enabled but would silently degrade:\n"
            + "\n".join(f"  - {problem}" for problem in readiness)
        )

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
        hard_negative_cfg=(config.get("data") or {}).get("hard_negative"),
    )
    # Report the buckets before refusing, so a failed launch still tells the
    # operator which bucket was empty and how many pairs each side had.
    bucket_summary = sampler.subtype_bucket_summary()
    _log_hard_negative_buckets(bucket_summary, rank)
    _require_non_degenerate_hard_negatives(bucket_summary)
    # The train probe is a fixed, reproducible, augmentation-free subset of
    # the train split evaluated with the exact validation evaluator.
    train_probe_dataset: Any = build_eval_dataset(
        train_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=probe_pairs_per_video,
        decoder=decoders["train"],
        group_by_negative_subtype=subtype_grouping,
    )
    val_quick_dataset: Any = build_eval_dataset(
        val_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        max_pairs_per_video=val_quick_pairs_per_video,
        decoder=decoders["val"],
        group_by_negative_subtype=subtype_grouping,
    )
    val_full_dataset: Any = build_eval_dataset(
        val_videos,
        test_delta,
        rank=rank,
        world_size=world_size,
        decoder=decoders["val"],
        group_by_negative_subtype=subtype_grouping,
    )
    test_full_dataset: Any = (
        build_eval_dataset(
            test_videos,
            test_delta,
            rank=rank,
            world_size=world_size,
            decoder=decoders["test"],
            group_by_negative_subtype=subtype_grouping,
        )
        if test_videos is not None
        else None
    )
    if subtype_grouping and "game_label_subtype" not in getattr(
        val_full_dataset, "group_catalogs", {}
    ):
        _warn_subtype_grouping_unavailable(
            rank,
            subtype_grouping,
            max_worst_subtype_fpr,
            "no video in the validation split carries a negative_subtype "
            "label (check data.metadata_sidecar)",
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
            f"Validation index does not contain legal delta={test_delta} pairs"
        )
    return LoaderBundle(
        train=DataLoader(train_dataset, batch_sampler=sampler, **train_common),
        train_probe=DataLoader(
            train_probe_dataset, batch_size=batch_size, **eval_common
        ),
        val_quick=DataLoader(val_quick_dataset, batch_size=batch_size, **eval_common),
        val_full=DataLoader(val_full_dataset, batch_size=batch_size, **eval_common),
        test_full=(
            DataLoader(test_full_dataset, batch_size=batch_size, **eval_common)
            if test_full_dataset is not None
            else None
        ),
        sampler=sampler,
        data_summary={
            "storage": "video_index_lazy_pairs",
            "image_backend": backend_name,
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "test_videos": (len(test_videos) if test_videos is not None else None),
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
