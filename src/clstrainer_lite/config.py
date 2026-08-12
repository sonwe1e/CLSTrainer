from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "run",
        "output_dir": "runs",
        "seed": 20260728,
    },
    "data": {
        # Switch only this field to choose the storage backend.
        "backend": "image",  # image | video
        "image": {
            "train_root": None,
            "test_root": None,
            "strict_filenames": True,
        },
        "video": {
            "train_root": None,
            "test_root": None,
            "extensions": [".mp4", ".mkv", ".mov", ".avi", ".m4v"],
            # Random-access training is chunk based.  The offline transcode tool
            # uses a short GOP so accurate seek + chunk decode stays cheap.
            "chunk_frames": 128,
            "cache_chunks": 2,
            "ffmpeg_bin": "ffmpeg",
            "ffprobe_bin": "ffprobe",
            "ffmpeg_threads": 2,
        },
        "val_ratio": 0.1,
        # Inclusive online train delta range.  [1,3] means each sample chooses
        # one of 1,2,3 at __getitem__ time; the index is NOT expanded 3x.
        "train_delta_range": [2, 2],
        # null => uniform over train_delta_range.  Otherwise length must equal
        # the number of integer deltas in the inclusive range.
        "train_delta_probabilities": None,
        # Validation/test remain fixed for comparable curves.
        "eval_delta": 2,
        "image_size": [208, 448],  # [height, width]
    },
    "augment": {
        "enabled": False,
        "horizontal_flip_p": 0.0,
        "crop_scale": [1.0, 1.0],
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "gamma": [1.0, 1.0],
        "color_shared": True,
        "noise_std": 0.0,
        "erase_p": 0.0,
        "erase_scale": [0.02, 0.10],
    },
    "loss": {
        "type": "cross_entropy",  # cross_entropy | focal
        "gamma": 1.5,
        "alpha": None,
    },
    "sampler": {
        "enabled": False,
        "class_probability": [0.5, 0.5],
        "game_balance_alpha": 0.5,
        # Global samples before DDP sharding. null keeps the natural dataset size.
        "samples_per_epoch": None,
    },
    "diagnostics": {
        "enabled": True,
        "histogram_bins": 20,
    },
    "model": {
        "factory": "clstrainer_lite.models:build_tiny_model",
        "kwargs": {},
        "init_weights": None,
    },
    "train": {
        "epochs": 10,
        "batch_size": 64,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": False,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "min_learning_rate": 1e-5,
        "gradient_clip_norm": 5.0,
        "decision_threshold": 0.5,
        "log_every_steps": 50,
        "n_val_step": 500,
        "n_test_step": 2000,
    },
    "runtime": {
        "accelerator": "auto",
        "backend": "auto",
        "amp": False,
        "amp_dtype": "bfloat16",
        "find_unused_parameters": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_override_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        dotted, raw = item.split("=", 1)
        cursor = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override nested key below {part!r}")
            cursor = child
        cursor[parts[-1]] = _parse_override_value(raw)
    return result


def _validate_range(name: str, value: Any, *, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [min, max]")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{name} requires min <= max")
    if positive and low <= 0:
        raise ValueError(f"{name} values must be positive")
    return low, high


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    augment = config["augment"]
    loss = config["loss"]
    sampler = config["sampler"]
    diagnostics = config["diagnostics"]
    train = config["train"]
    runtime = config["runtime"]

    storage_backend = str(data.get("backend", "image")).lower()
    if storage_backend not in {"image", "video"}:
        raise ValueError("data.backend must be image or video")
    backend_cfg = data.get(storage_backend, {}) or {}
    for key in ("train_root", "test_root"):
        if not backend_cfg.get(key):
            raise ValueError(f"data.{storage_backend}.{key} is required")

    val_ratio = float(data["val_ratio"])
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("data.val_ratio must be in (0, 1)")

    delta_range = data.get("train_delta_range")
    if not isinstance(delta_range, list) or len(delta_range) != 2:
        raise ValueError("data.train_delta_range must be [min_delta, max_delta]")
    delta_min, delta_max = (int(x) for x in delta_range)
    if delta_min <= 0 or delta_max <= 0 or delta_min > delta_max:
        raise ValueError("data.train_delta_range requires 0 < min_delta <= max_delta")
    probabilities = data.get("train_delta_probabilities")
    if probabilities is not None:
        expected = delta_max - delta_min + 1
        if not isinstance(probabilities, list) or len(probabilities) != expected:
            raise ValueError(
                f"data.train_delta_probabilities must be null or contain {expected} values"
            )
        if any(float(value) < 0 for value in probabilities) or sum(float(x) for x in probabilities) <= 0:
            raise ValueError("data.train_delta_probabilities must be non-negative and sum to > 0")
    if int(data["eval_delta"]) <= 0:
        raise ValueError("data.eval_delta must be positive")

    image_size = data.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("data.image_size must be [height, width]")
    if any(int(x) <= 0 for x in image_size):
        raise ValueError("data.image_size values must be positive")

    video = data.get("video", {}) or {}
    if int(video.get("chunk_frames", 128)) <= delta_max:
        raise ValueError("data.video.chunk_frames must be larger than max train delta")
    if int(video.get("cache_chunks", 2)) <= 0:
        raise ValueError("data.video.cache_chunks must be positive")
    if int(video.get("ffmpeg_threads", 2)) <= 0:
        raise ValueError("data.video.ffmpeg_threads must be positive")
    extensions = video.get("extensions", [])
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("data.video.extensions must be a non-empty list")

    if not 0.0 <= float(augment["horizontal_flip_p"]) <= 1.0:
        raise ValueError("augment.horizontal_flip_p must be in [0, 1]")
    _, crop_high = _validate_range("augment.crop_scale", augment["crop_scale"], positive=True)
    if crop_high > 1.0:
        raise ValueError("augment.crop_scale max must be <= 1")
    for key in ("brightness", "contrast", "saturation", "noise_std"):
        if float(augment[key]) < 0:
            raise ValueError(f"augment.{key} must be >= 0")
    if any(float(augment[key]) > 1.0 for key in ("brightness", "contrast", "saturation")):
        raise ValueError("augment brightness/contrast/saturation must be <= 1")
    _validate_range("augment.gamma", augment["gamma"], positive=True)
    if not 0.0 <= float(augment["erase_p"]) <= 1.0:
        raise ValueError("augment.erase_p must be in [0, 1]")
    _, erase_high = _validate_range("augment.erase_scale", augment["erase_scale"], positive=True)
    if erase_high > 1.0:
        raise ValueError("augment.erase_scale max must be <= 1")

    loss_type = str(loss["type"])
    if loss_type not in {"cross_entropy", "focal"}:
        raise ValueError("loss.type must be cross_entropy or focal")
    if float(loss.get("gamma", 1.5)) < 0:
        raise ValueError("loss.gamma must be >= 0")
    alpha = loss.get("alpha")
    if alpha is not None:
        if not isinstance(alpha, list) or len(alpha) != 2:
            raise ValueError("loss.alpha must be null or [class0_weight, class1_weight]")
        if any(float(x) <= 0 for x in alpha):
            raise ValueError("loss.alpha weights must be positive")

    class_probability = sampler.get("class_probability")
    if not isinstance(class_probability, list) or len(class_probability) != 2:
        raise ValueError("sampler.class_probability must be [class0, class1]")
    if any(float(x) < 0 for x in class_probability) or sum(float(x) for x in class_probability) <= 0:
        raise ValueError("sampler.class_probability must be non-negative and sum to > 0")
    game_alpha = float(sampler.get("game_balance_alpha", 0.5))
    if not 0.0 <= game_alpha <= 1.0:
        raise ValueError("sampler.game_balance_alpha must be in [0, 1]")
    samples_per_epoch = sampler.get("samples_per_epoch")
    if samples_per_epoch is not None and int(samples_per_epoch) <= 0:
        raise ValueError("sampler.samples_per_epoch must be null or positive")

    if int(diagnostics.get("histogram_bins", 20)) < 5:
        raise ValueError("diagnostics.histogram_bins must be >= 5")

    if int(train["epochs"]) <= 0:
        raise ValueError("train.epochs must be positive")
    if int(train["batch_size"]) <= 0:
        raise ValueError("train.batch_size must be positive")
    if int(train["num_workers"]) < 0:
        raise ValueError("train.num_workers must be >= 0")
    if float(train["learning_rate"]) <= 0:
        raise ValueError("train.learning_rate must be positive")
    if float(train["weight_decay"]) < 0:
        raise ValueError("train.weight_decay must be >= 0")
    if int(train["n_val_step"]) <= 0:
        raise ValueError("train.n_val_step must be positive")
    if int(train["n_test_step"]) <= 0:
        raise ValueError("train.n_test_step must be positive")
    if int(train.get("log_every_steps", 0)) < 0:
        raise ValueError("train.log_every_steps must be >= 0")
    threshold = float(train["decision_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("train.decision_threshold must be in (0, 1)")
    if str(train["scheduler"]) not in {"none", "cosine"}:
        raise ValueError("train.scheduler must be none or cosine")

    accelerator = str(runtime["accelerator"])
    if accelerator not in {"auto", "cpu", "cuda", "npu"}:
        raise ValueError("runtime.accelerator must be auto|cpu|cuda|npu")
    backend = str(runtime["backend"])
    if backend not in {"auto", "gloo", "nccl", "hccl"}:
        raise ValueError("runtime.backend must be auto|gloo|nccl|hccl")
    if str(runtime["amp_dtype"]) not in {"float16", "bfloat16"}:
        raise ValueError("runtime.amp_dtype must be float16 or bfloat16")
    if not isinstance(runtime.get("find_unused_parameters", True), bool):
        raise ValueError("runtime.find_unused_parameters must be true or false")


def _migrate_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep 0.2/0.3 configs usable while moving roots under image/video."""

    result = copy.deepcopy(raw)
    data = result.get("data")
    if not isinstance(data, dict):
        return result

    if "delta" in data:
        delta = int(data.pop("delta"))
        data.setdefault("train_delta_range", [delta, delta])
        data.setdefault("eval_delta", delta)

    # 0.3 used data.train_root / data.test_root.  By default they are PNG
    # roots; if the config explicitly says backend=video, treat them as video
    # roots instead.
    legacy_train = data.pop("train_root", None)
    legacy_test = data.pop("test_root", None)
    if legacy_train is not None or legacy_test is not None:
        backend = str(data.get("backend", "image")).lower()
        section = data.setdefault(backend, {})
        if legacy_train is not None:
            section.setdefault("train_root", legacy_train)
        if legacy_test is not None:
            section.setdefault("test_root", legacy_test)

    strict = data.pop("strict_filenames", None)
    if strict is not None:
        data.setdefault("image", {}).setdefault("strict_filenames", strict)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    raw = _migrate_legacy(raw)
    config = _deep_merge(DEFAULTS, raw)
    config = apply_overrides(config, list(overrides or []))

    # CLI compatibility: data.delta=N and legacy top-level roots.
    if "delta" in config["data"]:
        delta = int(config["data"].pop("delta"))
        config["data"]["train_delta_range"] = [delta, delta]
        config["data"]["eval_delta"] = delta
    backend = str(config["data"].get("backend", "image")).lower()
    for key in ("train_root", "test_root"):
        if key in config["data"]:
            config["data"].setdefault(backend, {})[key] = config["data"].pop(key)

    validate_config(config)
    # Read-only compatibility aliases for code that still inspects the 0.3
    # top-level root keys.  Dataset dispatch uses the backend-specific roots.
    active = config["data"][backend]
    config["data"]["train_root"] = active["train_root"]
    config["data"]["test_root"] = active["test_root"]
    if backend == "image":
        config["data"]["strict_filenames"] = config["data"]["image"]["strict_filenames"]
    return config
