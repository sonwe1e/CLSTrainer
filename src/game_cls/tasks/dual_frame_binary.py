from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.task import LossOutput, PredictionBatch, StepContext, TaskAdapter, TaskOutput
from ..registry import register


@dataclass(frozen=True)
class DualFrameBinaryTaskConfig:
    positive_class_index: int = 1
    num_classes: int = 2


class DualFrameBinaryTask:
    """Default task adapter for dual-frame binary classification.

    Wraps the existing batch validation, device transfer, forward, loss and
    prediction construction behind the :class:`TaskAdapter` interface
    (USERPLAN §5). It **must** call the existing ``combined_loss`` and
    ``ImageSpec.validate_pair_batch_shape`` rather than re-implementing them,
    so there is a single source of truth for the learning semantics.
    """

    task_name = "dual_frame_binary"
    contract_version = 1

    def __init__(self, image_spec: Any, loss_config: Any, task_config: Any | None = None) -> None:
        self.image_spec = image_spec
        self.loss_config = loss_config
        self.task_config = task_config or DualFrameBinaryTaskConfig()

    # --- validation -------------------------------------------------------
    def validate_model(self, model: Any) -> None:
        required = ("backbone", "cls")
        for attr in required:
            if not hasattr(model, attr):
                raise ValueError(
                    f"DualFrameBinaryTask expects the model to expose {required}; "
                    f"missing {attr!r}."
                )

    def validate_cpu_batch(self, batch: Any) -> None:
        images = batch["images"]
        if images.ndim != 5:
            raise ValueError(
                f"Expected a 5D image batch [B,2,C,H,W], got {images.ndim}D."
            )
        self.image_spec.validate_pair_batch_shape(images.shape)

    # --- device transfer --------------------------------------------------
    def move_batch_to_device(self, batch: Any, context: StepContext) -> Any:
        import torch

        images = batch["images"].to(context.device, non_blocking=True)
        labels = batch["labels"].to(context.device, non_blocking=True)
        if images.dtype == torch.uint8:
            compute_dtype = (
                torch.bfloat16
                if context.use_amp and context.amp_dtype == "bfloat16"
                else torch.float16
                if context.use_amp and context.amp_dtype == "float16"
                else torch.float32
            )
            images = images.to(compute_dtype).div_(255.0)
        return {
            "images": images,
            "labels": labels,
            "meta": batch.get("meta", [{}] * len(labels)),
        }

    # --- forward ---------------------------------------------------------
    def forward(self, model: Any, device_batch: Any, context: StepContext) -> TaskOutput:
        import torch

        del context
        images = device_batch["images"]
        logits = model(images[:, 0], images[:, 1])
        if logits.ndim != 2 or logits.shape[1] != self.task_config.num_classes:
            raise ValueError(
                f"Model must return [B,{self.task_config.num_classes}], "
                f"got {tuple(logits.shape)}"
            )
        return TaskOutput(raw=logits, extras={"logits_fp32": logits.float()})

    # --- loss ------------------------------------------------------------
    def compute_loss(
        self, output: TaskOutput, device_batch: Any, context: StepContext
    ) -> LossOutput:
        from ..losses.threshold_loss import combined_loss

        logits = output.raw
        labels = device_batch["labels"]
        total, components = combined_loss(
            logits, labels, self.loss_config, context.global_step, context.total_steps
        )
        return LossOutput(total=total, components=components)

    # --- predictions -----------------------------------------------------
    def build_predictions(
        self, output: TaskOutput, device_batch: Any, context: StepContext
    ) -> PredictionBatch:
        import torch

        del context
        logits = output.extras.get("logits_fp32", output.raw.float())
        labels = device_batch["labels"]
        positive_index = self.task_config.positive_class_index
        scores = torch.softmax(logits, dim=-1)[:, positive_index]
        margins = logits[:, positive_index] - logits[:, 1 - positive_index]
        predictions = torch.zeros_like(labels)
        return PredictionBatch(
            scores=scores,
            predictions=predictions,
            targets=labels,
            metadata=device_batch.get("meta", [{}] * len(labels)),
            extras={"logits": logits, "margins": margins},
        )


@register("task", "dual_frame_binary")
def build_task(config: Any, image_spec: Any) -> DualFrameBinaryTask:
    """Factory registered as ``game_cls.tasks.dual_frame_binary:build_task``.

    ``config`` is the ``task`` selector (with ``.params``); ``image_spec`` is
    the dataset image spec. The loss config is read from the top-level ``loss``
    section by the runner before calling this factory.
    """
    params = dict(config.params) if getattr(config, "params", None) else {}
    task_config = DualFrameBinaryTaskConfig(
        positive_class_index=int(params.get("positive_class_index", 1)),
        num_classes=int(params.get("num_classes", 2)),
    )
    return DualFrameBinaryTask(image_spec=image_spec, loss_config={}, task_config=task_config)
