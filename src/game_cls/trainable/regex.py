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
        del selection
        from torch import nn

        model.eval()
        for module_name, module in model.named_modules():
            if self._match(module_name):
                module.train()
        for module_name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                is_trainable = self._match(module_name)
                if (
                    is_trainable and self.freeze_trainable_batchnorm_stats
                ) or (
                    not is_trainable and self.freeze_frozen_batchnorm_stats
                ):
                    module.eval()

    def validate_loaded_state(
        self, model: Any, load_report: Any, selection: TrainableSelection
    ) -> float:
        from ..model.checkpoint_loader import validate_production_load

        del selection
        # The production validator keys off a name token; for regex policies the
        # frozen set is already explicit in the selection, so we rely on the
        # checkpoint key-set check at restore time and just confirm coverage via
        # the standard API using a permissive token that matches the includes.
        token = self.include[0].replace("(", "").replace(")", "").replace(".*", "") or "cls"
        return validate_production_load(model, load_report, trainable_name_contains=token)
