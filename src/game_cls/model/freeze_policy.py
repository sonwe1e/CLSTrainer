from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreezeSummary:
    trainable_names: tuple[str, ...]
    trainable_count: int
    frozen_count: int

    @property
    def trainable_ratio(self) -> float:
        total = self.trainable_count + self.frozen_count
        return self.trainable_count / total if total else 0.0


def configure_trainable_parameters(
    model, name_contains: str = "cls"
) -> FreezeSummary:
    trainable_names: list[str] = []
    trainable_count = 0
    frozen_count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name_contains in name
        count = parameter.numel()
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_count += count
        else:
            frozen_count += count
    if not trainable_names:
        raise RuntimeError(
            f"No trainable parameter contains the case-sensitive token {name_contains!r}."
        )
    return FreezeSummary(tuple(trainable_names), trainable_count, frozen_count)


def set_frozen_backbone_train_mode(
    model, name_contains: str = "cls", freeze_batchnorm_stats: bool = True
) -> None:
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Training mode configuration requires torch") from exc

    model.eval()
    for module_name, module in model.named_modules():
        if name_contains in module_name:
            module.train()
    if freeze_batchnorm_stats:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def assert_only_selected_parameters_changed(
    before: dict, model, name_contains: str = "cls"
) -> None:
    for name, parameter in model.named_parameters():
        changed = not parameter.detach().cpu().equal(before[name])
        if changed != (name_contains in name):
            expectation = "change" if name_contains in name else "remain unchanged"
            raise AssertionError(f"{name} was expected to {expectation}")


def snapshot_frozen_parameters(model) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def assert_frozen_parameters_unchanged(before: dict, model) -> None:
    current = dict(model.named_parameters())
    for name, expected in before.items():
        if name not in current:
            raise AssertionError(f"Frozen parameter disappeared: {name}")
        if not current[name].detach().cpu().equal(expected):
            raise AssertionError(f"Frozen parameter changed during training: {name}")
