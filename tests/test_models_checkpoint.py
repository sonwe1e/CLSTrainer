from pathlib import Path

import torch

from clstrainer_lite.checkpoint import save_model_weights
from clstrainer_lite.models import build_tiny_model


def test_save_pure_model_state(tmp_path: Path):
    model = build_tiny_model(width=4)
    path = tmp_path / "model.pt"
    save_model_weights(model, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == set(model.state_dict())
