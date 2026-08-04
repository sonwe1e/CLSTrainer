from __future__ import annotations

import importlib
from typing import Any, Callable


def resolve_factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("model.factory must use package.module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Model factory is not callable: {spec}")
    return factory


def build_model(config: dict[str, Any]):
    factory = resolve_factory(config["factory"])
    model = factory(config)
    return model


def build_demo_model(config: dict[str, Any] | None = None):
    import torch
    from torch import nn

    num_classes = int((config or {}).get("num_classes", 2))

    class DemoDualFrameModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(6, 6, kernel_size=1, groups=6, bias=False),
                nn.AdaptiveAvgPool2d(1),
            )
            with torch.no_grad():
                self.backbone[0].weight.fill_(1.0)
            self.cls = nn.Linear(6, num_classes)

        def forward(self, image0, image1):
            features = self.backbone(torch.cat([image0, image1], dim=1)).flatten(1)
            return self.cls(features)

    return DemoDualFrameModel()
