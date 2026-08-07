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


def configure_trainable_parameters(model, name_contains: str = "cls") -> FreezeSummary:
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
    model,
    name_contains: str = "cls",
    freeze_batchnorm_stats: bool | None = None,
    *,
    freeze_backbone_batchnorm_stats: bool = True,
    freeze_cls_batchnorm_stats: bool = True,
) -> None:
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Training mode configuration requires torch") from exc

    model.eval()
    # Put every module on a path to a trainable parameter into train mode. The
    # legacy test was ``name_contains in module_name`` (the "cls" token), which
    # left staged-unfreeze stages -- whose parameter names carry no "cls" token
    # (e.g. ``layer4`` / ``backbone.stage4``) -- in eval mode while their
    # weights were being updated: Dropout/DropPath never ran and the freshly
    # unfrozen weights trained under inference semantics (audit P1-2). The
    # ``requires_grad`` set reflects the current trainable rules, so it covers
    # both the legacy token and the staged path.
    trainable_subtrees = {
        module_name
        for module_name, module in model.named_modules()
        if any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=True)
        )
    }
    for module_name, module in model.named_modules():
        if module_name in trainable_subtrees:
            module.train()
    # ``module.train()`` on a mixed container (e.g. the model root) recursively
    # trains its frozen subtrees too; put any subtree that holds only frozen
    # parameters back into eval so its Dropout/DropPath/stochastic-depth stay
    # deterministic and match the frozen-parameter contract.
    for _, module in model.named_modules():
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(
            parameter.requires_grad for parameter in parameters
        ):
            module.eval()
    if freeze_batchnorm_stats is not None:
        freeze_backbone_batchnorm_stats = freeze_batchnorm_stats
        freeze_cls_batchnorm_stats = freeze_batchnorm_stats
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            is_cls = name_contains in module_name
            if (is_cls and freeze_cls_batchnorm_stats) or (
                not is_cls and freeze_backbone_batchnorm_stats
            ):
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
