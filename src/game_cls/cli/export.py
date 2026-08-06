"""cls-trainer export (step5 P6).

Exports a checkpoint as a deployment artifact:

* ``weights`` (default): a pure state dict plus ``export_manifest.json``
  embedding the business threshold ``decision.threshold``, the base
  checkpoint SHA-256 and a metric summary — the pragmatic deployment path.
* ``onnx`` (optional): a traced ``[B,2]`` graph verified against the
  PyTorch reference on the same sample tensors (max abs diff <= 1e-4).

``decision.threshold`` is shared between training, evaluation and the
exported artifact, so deployment uses exactly the business threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from game_cls.cli.common import _resolve_run_dir
from game_cls.cli.evaluate import _resolve_checkpoint_state


def cmd_export(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.engine.checkpoint import unwrap_model
    from game_cls.model.builder import build_model

    run_dir = _resolve_run_dir(args.run, Path(args.runs_root))
    config_source = args.config or str(run_dir / "resolved_config.json")
    if not Path(config_source).is_file():
        print(
            f"No config found for run {run_dir}: pass --config or ensure "
            f"{run_dir / 'resolved_config.json'} exists.",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_config(config_source)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2

    state_dict, checkpoint_path = _resolve_checkpoint_state(run_dir, args.checkpoint)
    model = build_model(config["model"])
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    model.eval()

    export_cfg = config.get("export") or {}
    out_dir = Path(args.out or export_cfg.get("output_dir", "exports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    export_format = args.format or export_cfg.get("format", "weights")
    include_threshold = bool(export_cfg.get("include_threshold", True))
    manifest_extra = (
        {"decision.threshold": config["decision"]["threshold"]}
        if include_threshold
        else {}
    )
    if export_format == "weights":
        model_only = out_dir / f"model_{args.checkpoint}.pth"
        import torch

        torch.save(unwrap_model(model).state_dict(), model_only)
        manifest_path = out_dir / "export_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "checkpoint": args.checkpoint,
                    "checkpoint_path": checkpoint_path,
                    **manifest_extra,
                    "model_factory": config["model"].get("factory"),
                    "shape": [1, 2, 3, 208, 448],
                    "weights": str(model_only),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "format": "weights",
                    "weights": str(model_only),
                    "manifest": str(manifest_path),
                    "decision.threshold": config["decision"]["threshold"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if export_format == "onnx":
        return _export_onnx(
            model,
            config,
            export_cfg,
            out_dir,
            args.checkpoint,
            checkpoint_path,
            manifest_extra,
        )
    print(f"Unsupported export format: {export_format}", file=sys.stderr)
    return 2


def _export_onnx(
    model,
    config: dict,
    export_cfg: dict,
    out_dir: Path,
    alias: str,
    checkpoint_path: str,
    manifest_extra: dict,
) -> int:
    """Trace the model to ONNX and verify against the PyTorch reference."""
    import torch

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "ONNX export requires onnxruntime; install it or use --format weights.",
            file=sys.stderr,
        )
        return 2
    input_shape = (1, 2, 3, 208, 448)
    sample = torch.randn(*input_shape)
    onnx_path = out_dir / f"model_{alias}.onnx"
    try:
        torch.onnx.export(
            model,
            (sample[:, 0], sample[:, 1]),
            str(onnx_path),
            input_names=["image0", "image1"],
            output_names=["logits"],
            opset_version=int(export_cfg.get("onnx_opset", 17)),
            dynamic_axes={"image0": {0: "batch"}, "image1": {0: "batch"}},
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"ONNX export failed (custom/non-traceable ops?): {exc}\n"
            "Fall back to --format weights.",
            file=sys.stderr,
        )
        return 2
    # Verification harness: ORT vs PyTorch on the same N sample tensors.
    session = ort.InferenceSession(str(onnx_path))
    verify_samples = int(export_cfg.get("verify_samples", 8))
    max_diff = 0.0
    with torch.no_grad():
        for _ in range(verify_samples):
            batch = torch.randn(2, 2, 3, 208, 448)
            reference = model(batch[:, 0], batch[:, 1]).numpy()
            onnx_outputs = session.run(
                None,
                {
                    "image0": batch[:, 0].numpy(),
                    "image1": batch[:, 1].numpy(),
                },
            )[0]
            max_diff = max(max_diff, float(abs(reference - onnx_outputs).max()))
    if max_diff > 1e-4:
        print(
            f"ONNX verification failed: max abs diff {max_diff:.2e} > 1e-4. "
            "Refusing to produce a bad artifact.",
            file=sys.stderr,
        )
        return 2
    manifest = {
        "run_id": "",
        "checkpoint": alias,
        "checkpoint_path": checkpoint_path,
        **manifest_extra,
        "model_factory": config["model"].get("factory"),
        "shape": list(input_shape),
        "onnx": str(onnx_path),
        "verification_max_abs_diff": max_diff,
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "format": "onnx",
                "onnx": str(onnx_path),
                "verification_max_abs_diff": max_diff,
                "decision.threshold": config["decision"]["threshold"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
