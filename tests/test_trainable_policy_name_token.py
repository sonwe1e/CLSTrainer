from __future__ import annotations

import torch

from game_cls.model.builder import build_demo_model
from game_cls.model.freeze_policy import configure_trainable_parameters
from game_cls.trainable import NameTokenTrainablePolicy


def test_name_token_matches_legacy_selection() -> None:
    torch.manual_seed(3)
    model_legacy = build_demo_model({})
    legacy = configure_trainable_parameters(model_legacy, "cls")
    legacy_trainable = set(legacy.trainable_names)

    torch.manual_seed(3)
    model_new = build_demo_model({})
    policy = NameTokenTrainablePolicy(token="cls")
    selection = policy.select(model_new)

    assert set(selection.trainable_state.parameter_keys) == legacy_trainable
    assert set(selection.frozen_state.parameter_keys) == (
        set(dict.fromkeys(n for n, _ in model_new.named_parameters())) - legacy_trainable
    )
    assert len(selection.groups) == 1
    assert selection.groups[0].name == "head"


def test_name_token_no_match_raises() -> None:
    model = build_demo_model({})
    policy = NameTokenTrainablePolicy(token="nonexistent_token")
    try:
        policy.select(model)
    except RuntimeError as exc:
        assert "nonexistent_token" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for no-match token")
