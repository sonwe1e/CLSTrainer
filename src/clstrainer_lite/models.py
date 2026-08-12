from __future__ import annotations

import importlib
from typing import Any, Callable

import torch
from torch import nn


def resolve_factory(spec: str) -> Callable[..., nn.Module]:
    if ":" not in spec:
        raise ValueError("model.factory must use package.module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Model factory is not callable: {spec}")
    return factory


def build_model(config: dict[str, Any]) -> nn.Module:
    factory = resolve_factory(str(config["factory"]))
    kwargs = dict(config.get("kwargs") or {})
    model = factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("model.factory must return torch.nn.Module")
    return model


def load_initial_weights(model: nn.Module, path: str | None) -> None:
    if not path:
        return
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Initial weights must be a state_dict: {path}")
    state = {str(key).removeprefix("module."): value for key, value in payload.items()}
    model.load_state_dict(state, strict=True)


class TinyPairCNN(nn.Module):
    """Small built-in model for smoke tests and as an editable example."""

    def __init__(self, channels: int = 3, width: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        in_channels = channels * 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> torch.Tensor:
        x = torch.cat((image0, image1), dim=1)
        x = self.features(x).flatten(1)
        return self.classifier(x)


def build_tiny_model(channels: int = 3, width: int = 32, num_classes: int = 2) -> nn.Module:
    return TinyPairCNN(channels=channels, width=width, num_classes=num_classes)
