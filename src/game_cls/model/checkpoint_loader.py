from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]


def extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError("Checkpoint must be a state_dict or contain model/state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_model_checkpoint(model, path: str | Path) -> LoadReport:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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

