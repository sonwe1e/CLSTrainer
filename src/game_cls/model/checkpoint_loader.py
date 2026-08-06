from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]


def validate_production_load(
    model,
    report: LoadReport,
    *,
    trainable_name_contains: str = "cls",
    frozen_parameter_names: set[str] | None = None,
) -> float:
    state_keys = set(model.state_dict())
    if frozen_parameter_names is not None:
        # Rule-based training: the frozen set is whatever the rules leave
        # frozen at the current step, not the legacy name token.
        frozen_keys = {key for key in state_keys if key in frozen_parameter_names}
    else:
        frozen_keys = {key for key in state_keys if trainable_name_contains not in key}
    loaded_frozen = {key for key in report.loaded if key in frozen_keys}
    missing_frozen = {key for key in report.missing if key in frozen_keys}
    mismatch_frozen = {key for key in report.shape_mismatch if key in frozen_keys}
    coverage = len(loaded_frozen) / len(frozen_keys) if frozen_keys else 1.0
    if missing_frozen or mismatch_frozen or coverage < 1.0:
        raise RuntimeError(
            "Production checkpoint must load 100% of the frozen backbone "
            "parameters and buffers: "
            f"coverage={coverage:.2%}, "
            f"missing_frozen={sorted(missing_frozen)}, "
            f"shape_mismatch_frozen={sorted(mismatch_frozen)}"
        )
    return coverage


def extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("state_dict"), dict
    ):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError("Checkpoint must be a state_dict or contain model/state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model_checkpoint(model, path: str | Path) -> LoadReport:
    import torch

    # Base-model checkpoints must be plain tensor state dicts (or a dict
    # containing one). weights_only=True refuses pickled code execution,
    # so untrusted .pth files cannot run arbitrary code at load time.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Base checkpoint {path} could not be loaded with "
            "weights_only=True; it must contain only tensors and plain "
            "dicts (a pickle with embedded code is refused). "
            f"Original error: {exc}"
        ) from exc
    incoming = extract_state_dict(checkpoint)
    current = model.state_dict()
    compatible = {}
    shape_mismatch = []
    for key, value in incoming.items():
        if key in current and tuple(value.shape) != tuple(current[key].shape):
            shape_mismatch.append(key)
        elif key in current:
            compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    return LoadReport(
        loaded=tuple(sorted(compatible)),
        missing=tuple(sorted(result.missing_keys)),
        unexpected=tuple(sorted(key for key in incoming if key not in current)),
        shape_mismatch=tuple(sorted(shape_mismatch)),
    )
