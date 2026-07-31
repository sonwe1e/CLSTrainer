from __future__ import annotations

from typing import Any

from ..contracts.trainable import StateSelection, TrainableSelection
from .base import TrainablePolicyBase, _group_spec


class NameTokenTrainablePolicy(TrainablePolicyBase):
    """Selects trainable parameters by a case-sensitive name token (default ``cls``).

    This policy **exactly** reproduces the legacy behavior implemented in
    :mod:`game_cls.model.freeze_policy`: a parameter is trainable iff the token
    appears in its name. BatchNorm running stats are frozen per configuration.
    """

    policy_name = "name_token"

    def __init__(
        self,
        token: str = "cls",
        *,
        case_sensitive: bool = True,
        freeze_trainable_batchnorm_stats: bool = True,
        freeze_frozen_batchnorm_stats: bool = True,
    ) -> None:
        self.token = token
        self.case_sensitive = case_sensitive
        self.freeze_trainable_batchnorm_stats = freeze_trainable_batchnorm_stats
        self.freeze_frozen_batchnorm_stats = freeze_frozen_batchnorm_stats

    def _match(self, name: str) -> bool:
        if self.case_sensitive:
            return self.token in name
        return self.token.casefold() in name.casefold()

    def select(self, model: Any) -> TrainableSelection:
        trainable_names: list[str] = []
        for name, parameter in model.named_parameters():
            if self._match(name):
                parameter.requires_grad = True
                trainable_names.append(name)
            else:
                parameter.requires_grad = False
        if not trainable_names:
            raise RuntimeError(
                f"No trainable parameter contains the token {self.token!r} "
                f"(case_sensitive={self.case_sensitive})."
            )
        all_names = list(dict.fromkeys(name for name, _ in model.named_parameters()))
        frozen_names = [name for name in all_names if name not in set(trainable_names)]

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
        # Derive module modes from the selection rather than re-inferring
        # from the token. This ensures case_sensitive=False is respected
        # and the behavior is consistent with select().
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
        # Apply BatchNorm freezing based on whether the module is trainable.
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
        # Validate that the frozen backbone parameters are fully covered by
        # the loaded checkpoint. The frozen set is explicit in the selection.
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
