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
        trainable_set = set(trainable_names)
        all_param_names = list(dict.fromkeys(name for name, _ in model.named_parameters()))
        frozen_names = [name for name in all_param_names if name not in trainable_set]

        # Classify persistent buffers: these are state_dict keys that are NOT
        # parameters but ARE persisted (e.g. BatchNorm running_mean, running_var,
        # num_batches_tracked). Non-persistent buffers (e.g. _version counters)
        # are excluded. Including persistent buffers in the frozen coverage
        # validation ensures a frozen backbone's BN stats are fully protected.
        parameter_name_set = set(all_param_names)
        persistent_buffer_names = {
            key
            for key in model.state_dict()
            if key not in parameter_name_set
            and not key.startswith("_")
        }
        # ``persistent_buffers()`` is the authoritative source for which buffers
        # are persisted in state_dict. Fall back to the state_dict-based heuristic
        # only when it is unavailable.
        try:
            persistent_buffer_names = set(
                name for name, _ in model.named_buffers(recurse=True)
                if name in model.state_dict()
            )
        except Exception:
            pass

        trainable_buffer_names = {
            name for name in persistent_buffer_names if self._match(name)
        }
        frozen_buffer_names = list(persistent_buffer_names - trainable_buffer_names)

        head = _group_spec("head", trainable_names, lr_multiplier=1.0)
        return TrainableSelection(
            groups=(head,),
            frozen_parameter_names=tuple(frozen_names),
            trainable_state=StateSelection(
                parameter_keys=tuple(trainable_names),
                buffer_keys=tuple(sorted(trainable_buffer_names)),
            ),
            frozen_state=StateSelection(
                parameter_keys=tuple(frozen_names),
                buffer_keys=tuple(frozen_buffer_names),
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
        # Validate that the frozen backbone parameters AND persistent buffers
        # (e.g. BatchNorm running_mean, running_var, num_batches_tracked) are
        # fully covered by the loaded checkpoint. The frozen set is explicit in
        # the selection.
        frozen_keys = set(selection.frozen_state.parameter_keys) | set(
            selection.frozen_state.buffer_keys
        )
        if not frozen_keys:
            return 1.0
        loaded = set(getattr(load_report, "loaded", ()))
        missing = frozen_keys - loaded
        if missing:
            raise RuntimeError(
                "Production checkpoint must load 100% of the frozen backbone "
                "(parameters + persistent buffers). "
                f"Missing {len(missing)} frozen state(s): "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return 1.0
