from __future__ import annotations

import copy

import torch

from game_cls.contracts.task import StepContext
from game_cls.data.image_spec import ImageSpec
from game_cls.model.builder import build_demo_model
from game_cls.model.freeze_policy import (
    assert_frozen_parameters_unchanged,
    configure_trainable_parameters,
    snapshot_frozen_parameters,
)
from game_cls.tasks import DualFrameBinaryTask


def _make_batch(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 96, (4, 2, 3, 208, 448), dtype=torch.uint8, generator=generator)
    labels = torch.tensor([0, 1, 0, 1])
    return {"images": images, "labels": labels, "meta": [{}] * 4}


def _loss_config() -> dict:
    return {
        "threshold": 0.99,
        "threshold_loss_weight": 0.2,
        "threshold_safety_margin": 0.2,
        "threshold_temperature": 0.5,
        "threshold_warmup_ratio": 0.1,
        "threshold_ramp_ratio": 0.2,
        "cross_entropy_weight": 1.0,
    }


def test_task_adapter_matches_legacy_10step_update() -> None:
    """The TaskAdapter must produce the same 10-step parameter update as the
    legacy manual hot path (USERPLAN §5.5)."""
    spec = ImageSpec(width=448, height=208, channels=3)
    loss_config = _loss_config()

    torch.manual_seed(3)
    model_legacy = build_demo_model({})
    configure_trainable_parameters(model_legacy, "cls")

    torch.manual_seed(3)
    model_task = build_demo_model({})
    task = DualFrameBinaryTask(image_spec=spec, loss_config=loss_config)
    configure_trainable_parameters(model_task, "cls")

    frozen = snapshot_frozen_parameters(model_task)
    optimizer_legacy = torch.optim.AdamW(
        [p for p in model_legacy.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer_task = torch.optim.AdamW(
        [p for p in model_task.parameters() if p.requires_grad], lr=1e-3
    )

    ctx = StepContext(
        global_step=0, total_steps=100, epoch=0,
        device=torch.device("cpu"), use_amp=False, amp_dtype="float16",
    )
    for step in range(10):
        batch = _make_batch(seed=step)
        # Legacy manual path.
        imgs = batch["images"].float().div_(255.0)
        logits = model_legacy(imgs[:, 0], imgs[:, 1])
        from game_cls.losses.threshold_loss import combined_loss

        loss_legacy, _ = combined_loss(logits, batch["labels"], loss_config, step, 100)
        optimizer_legacy.zero_grad(set_to_none=True)
        loss_legacy.backward()
        optimizer_legacy.step()

        # Task adapter path.
        device_batch = task.move_batch_to_device(batch, ctx)
        task_output = task.forward(model_task, device_batch, ctx)
        loss_output = task.compute_loss(task_output, device_batch, ctx)
        optimizer_task.zero_grad(set_to_none=True)
        loss_output.total.backward()
        optimizer_task.step()
        ctx = StepContext(
            global_step=step + 1, total_steps=100, epoch=0,
            device=torch.device("cpu"), use_amp=False, amp_dtype="float16",
        )

    for p_legacy, p_task in zip(model_legacy.parameters(), model_task.parameters()):
        assert torch.equal(p_legacy, p_task), "10-step parameter update diverged"
    assert_frozen_parameters_unchanged(frozen, model_task)
