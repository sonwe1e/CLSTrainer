from __future__ import annotations

from typing import Any


def cross_validate(config: dict[str, Any]) -> None:
    """Run cross-field checks that span multiple config sections.

    These are the checks the USERPLAN lists as "跨字段校验". They run after
    schema validation so every key is known to exist.
    """
    runtime = config.get("runtime", {})
    accelerator = str(runtime.get("accelerator", {}).get("type", "cpu"))
    distributed = runtime.get("distributed", {})
    distributed_type = str(distributed.get("type", "single_process"))

    threshold = _decision_threshold(config)
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        raise ValueError(f"evaluation.decision.threshold must be in [0,1], got {threshold}")

    if distributed_type == "ddp":
        import os

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            raise RuntimeError(
                "runtime.distributed.type=ddp requires WORLD_SIZE > 1."
            )


def _decision_threshold(config: dict[str, Any]) -> float | None:
    params = config.get("evaluation", {}).get("decision", {}).get("params", {})
    value = params.get("threshold")
    return float(value) if value is not None else None
