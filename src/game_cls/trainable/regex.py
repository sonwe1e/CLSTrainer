from __future__ import annotations

import re
from typing import Any

from ..contracts.trainable import StateSelection, TrainableSelection
from .base import TrainablePolicyBase, _group_spec


class RegexTrainablePolicy(TrainablePolicyBase):
    """Selects trainable parameters via include/exclude regex patterns.

    A parameter is trainable when it matches at least one ``include`` pattern and
    no ``exclude`` pattern. This supports LoRA-style selection (e.g. ``\\.lora_$``)
    without hard-coding the parameter names (USERPLAN §7.4).
    """

    policy_name = "regex"

    def __init__(
        self,
        *,
        include: list[str],
        exclude: list[str] | None = None,
        freeze_trainable_batchnorm_stats: bool = True,
        freeze_frozen_batchnorm_stats: bool = True,
    ) -> None:
        self.include = list(include)
        self.exclude = list(exclude or [])
        self.freeze_trainable_batchnorm_stats = freeze_trainable_batchnorm_stats
        self.freeze_frozen_batchnorm_stats = freeze_frozen_batchnorm_stats
        self._include = [re.compile(pattern) for pattern in self.include]
        self._exclude = [re.compile(pattern) for pattern in self.exclude]

    def _match(self, name: str) -> bool:
        included = any(pattern.search(name) for pattern in self._include)
        if not included:
            return False
        return not any(pattern.search(name) for pattern in self._exclude)

    def select(self, model: Any) -> TrainableSelection:
        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, parameter in model.named_parameters():
            if self._match(name):
                parameter.requires_grad = True
                trainable_names.append(name)
            else:
                parameter.requires_grad = False
                frozen_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                "RegexTrainablePolicy matched no parameters. "
                f"include={self.include} exclude={self.exclude}"
            )
        head = _group_spec("head", trainable_names, lr_multiplier=1.0)
        return TrainableSelection(
            groups=(head,),
            frozen_parameter_names=tuple(frozen_names),
            trainable_state=StateSelection(
                parameter_keys=tuple(trainable_names), buffer_keys=()
            ),
            frozen_state=StateSelection(
                parameter_keys=tuple(frozen_names), buffer_keys=()
            ),
        )

    def configure_module_modes(self, model: Any, selection: TrainableSelection) -> None:
        # Derive module names from selected parameters rather than applying
        # the parameter regex to module names. A regex like `\.lora_[AB]$`
        # matches parameter names (e.g. ``encoder.block.attn.lora_A``) but not
        # module names (e.g. ``encoder.block.attn``).
        trainable_param_names = set(selection.trainable_state.parameter_keys)
        trainable_module_names = {
            name.rsplit(".", 1)[0] if "." in name else name
            for name in trainable_param_names
        }
        from torch import nn

        model.eval()
        for module_name, module in model.named_modules():
            if module_name in trainable_module_names:
                module.train()
        for module_name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                is_trainable = module_name in trainable_module_names
                if (
                    is_trainable and self.freeze_trainable_batchnorm_stats
                ) or (
                    not is_trainable and self.freeze_frozen_batchnorm_stats
                ):
                    module.eval()

    def validate_loaded_state(
        self, model: Any, load_report: Any, selection: TrainableSelection
    ) -> float:
        # Validate that the frozen backbone parameters are fully covered by the
        # loaded checkpoint. For regex policies the frozen set is explicit in
        # the selection, so we check coverage directly against those keys rather
        # than trying to reverse a regex into a token.
        frozen_keys = set(selection.frozen_state.parameter_keys)
        if not frozen_keys:
            return 1.0
        loaded = set(getattr(load_report, "loaded", ()))
        missing = frozen_keys - loaded
        if missing:
            raise RuntimeError(
                "Production checkpoint must load 100% of the frozen backbone. "
                f"Missing {len(missing)} frozen parameter(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return 1.0
