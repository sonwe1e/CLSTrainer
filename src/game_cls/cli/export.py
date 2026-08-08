"""cls-trainer export (step5 P6).

Exports a checkpoint as a deployment artifact:

* ``weights`` (default): a pure state dict plus ``export_manifest.json``
  embedding the business threshold ``decision.threshold``, the base
  checkpoint SHA-256 and a metric summary — the pragmatic deployment path.
* ``onnx`` (optional): a traced ``[B,2]`` graph verified against the
  PyTorch reference on the same sample tensors (max abs diff <= 1e-4).

``decision.threshold`` is shared between training, evaluation and the
exported artifact, so deployment uses exactly the business threshold.

The declared input shape is derived from ``data.channels/height/width``
rather than hardcoded, so overriding the task profile cannot desynchronize
the manifest (and the traced graph) from what the model was trained on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from game_cls.cli.common import _resolve_run_dir
from game_cls.cli.evaluate import _resolve_checkpoint_state
from game_cls.contract import CONTRACT_VERSION

# The pair dimension is an architectural constant, not a config key: the
# model contract is ``forward(image0, image1) -> [B,2]`` and the whole
# pipeline hands it uint8 ``[B,2,C,H,W]`` frame pairs (see
# ImageSpec.validate_pair_batch_shape). Only C/H/W are configurable.
FRAME_PAIR_SIZE = 2


def _cleanup_empty_export_parents(artifact_dir: Path, export_root: Path) -> None:
    """Remove empty hash/run directories left by a failed transaction."""
    parent = artifact_dir.parent
    while parent != export_root and export_root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


# Deployment-relevant headline metrics copied into the manifest. The full
# metric set stays in the run's training_summary.json (referenced by path in
# metric_summary["source"]); a deployment manifest only needs the numbers an
# operator gates a release on.
_SUMMARY_METRIC_KEYS = (
    "sample_count",
    "threshold",
    "global_f1_at_decision_threshold",
    "macro_game_f1_at_decision_threshold",
    "worst_game_f1_at_decision_threshold",
    "global_fpr_at_decision_threshold",
    "global_positive_recall_at_decision_threshold",
    "worst_game_fpr_at_decision_threshold",
    "worst_game_positive_recall_at_decision_threshold",
    "worst_subtype_fpr_at_decision_threshold",
    "negative_score_p99",
    "negative_score_p999",
)


def _input_shape(config: dict) -> tuple[int, int, int, int, int]:
    """``(1, 2, C, H, W)`` from the resolved config, never hardcoded."""
    from game_cls.data.image_spec import ImageSpec

    spec = ImageSpec.from_config(config["data"])
    return (1, FRAME_PAIR_SIZE, *spec.chw)


def _headline_metrics(metrics: Any) -> dict[str, Any]:
    """Pick the headline scalars out of one evaluation metrics dict."""
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics[key]
        for key in _SUMMARY_METRIC_KEYS
        if metrics.get(key) is not None
    }


def _metric_summary(run_dir: Path, config: dict) -> dict[str, Any]:
    """Summarize the run's training_summary.json for the manifest.

    Exports happen long after training, so the numbers are read back from
    the run directory rather than recomputed; a run without a summary (an
    export of a hand-placed checkpoint) yields an empty summary instead of
    an error.
    """
    summary_path = run_dir / "training_summary.json"
    summary: dict[str, Any] = {
        "source": str(summary_path),
    }
    if not summary_path.is_file():
        summary["available"] = False
        return summary
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        summary["available"] = False
        return summary
    # P1 (export manifest): prefer the annotated selection_metric from the
    # checkpoint's best-selection record so constrained-selection runs don't
    # fall back to the config default "global_f1_at_decision_threshold".
    _best_sel = payload.get("best_selection")
    selection_metric = (
        _best_sel.get("selection_metric") if isinstance(_best_sel, dict) else None
    ) or (config.get("evaluation") or {}).get(
        "selection_metric", "global_f1_at_decision_threshold"
    )
    summary["selection_metric"] = selection_metric
    summary["selection_mode"] = (
        _best_sel.get("selection_mode") if isinstance(_best_sel, dict) else None
    ) or (config.get("evaluation") or {}).get("selection_mode", "metric")
    summary["selection_eligible"] = (
        _best_sel.get("selection_eligible") if isinstance(_best_sel, dict) else None
    )
    best = payload.get("best_validation_metrics")
    summary["available"] = True
    summary["global_step"] = payload.get("global_step")
    summary["selection_metric_value"] = (
        best.get(selection_metric) if isinstance(best, dict) else None
    )
    summary["best_validation"] = _headline_metrics(best)
    summary["last_checkpoint"] = _headline_metrics(
        payload.get("last_checkpoint_metrics")
    )
    return summary


def _base_checkpoint_sha256(run_dir: Path, config: dict) -> str | None:
    """SHA-256 of the pretrained backbone the run started from.

    The training run already hashed it into manifest.json, so that value is
    authoritative (the file may have moved or changed since). Only fall back
    to hashing ``model.checkpoint_path`` when the run has no manifest entry.
    """
    from game_cls.runs import read_manifest, sha256_file

    manifest = read_manifest(run_dir) or {}
    recorded = manifest.get("base_checkpoint_sha256")
    if recorded:
        return str(recorded)
    base_checkpoint = (config.get("model") or {}).get("checkpoint_path")
    if not base_checkpoint or not Path(base_checkpoint).is_file():
        return None
    return sha256_file(base_checkpoint)


def _export_manifest(
    *,
    run_dir: Path,
    config: dict,
    alias: str,
    checkpoint_path: str,
    manifest_extra: dict,
    input_shape: tuple[int, ...],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Build the manifest shared by the weights and ONNX paths."""
    _ms = _metric_summary(run_dir, config)
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_dir.name,
        "checkpoint": alias,
        "checkpoint_path": checkpoint_path,
        **manifest_extra,
        "base_checkpoint": (config.get("model") or {}).get("checkpoint_path"),
        "base_checkpoint_sha256": _base_checkpoint_sha256(run_dir, config),
        "model_factory": config["model"].get("factory"),
        "shape": list(input_shape),
        **artifact,
        "metric_summary": _ms,
        # P1 (export manifest): surface selection semantics so deployment tools
        # can read why this checkpoint was selected without re-running training.
        "selection_mode": _ms.get("selection_mode"),
        "selection_eligible": _ms.get("selection_eligible"),
    }


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

    from game_cls.release import ReleaseVerificationError, verify_release_artifact

    try:
        verified = verify_release_artifact(run_dir, args.checkpoint, config)
    except ReleaseVerificationError as exc:
        print(f"Export refused: {exc}", file=sys.stderr)
        return 2
    state_dict, checkpoint_path = _resolve_checkpoint_state(run_dir, args.checkpoint)
    model = build_model(config["model"])
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    model.eval()

    export_cfg = config.get("export") or {}
    # argparse leaves --out/--format as None so the config actually decides;
    # the literal fallbacks only matter for a raw config dict that never went
    # through finalize_config (which fills both keys).
    out_dir = Path(args.out or export_cfg.get("output_dir") or "exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    from game_cls.reports.benchmark import file_sha256

    # Audit PR-F: immutable per-artifact layout
    #   exports/<run_id>/<checkpoint_sha256>/<format>/artifact + manifest
    # A second export of the same artifact writes the same directory and is
    # refused; a different checkpoint writes a different sha directory, so runs
    # can never overwrite each other's deployment artifacts.
    export_format = args.format or export_cfg.get("format") or "weights"
    final_artifact_dir = (
        out_dir / run_dir.name / file_sha256(checkpoint_path) / export_format
    )
    if final_artifact_dir.exists():
        print(
            f"Export refused: artifact already exists at {final_artifact_dir}; the "
            "export layout is immutable (audit PR-F). Delete it to re-export.",
            file=sys.stderr,
        )
        return 2
    final_artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = final_artifact_dir.with_name(
        f".{final_artifact_dir.name}.staging-{uuid4().hex}"
    )
    artifact_dir.mkdir(parents=False, exist_ok=False)
    include_threshold = bool(export_cfg.get("include_threshold", True))
    manifest_extra = (
        {"decision.threshold": config["decision"]["threshold"]}
        if include_threshold
        else {}
    )
    manifest_extra.update(
        {
            "release_identity_sha256": verified.release_identity_sha256,
            "release_identity": verified.identity,
            "benchmark_report": str(verified.report_path),
        }
    )
    input_shape = _input_shape(config)
    if export_format == "weights":
        model_only = artifact_dir / "model.pt"
        import torch

        manifest_path = artifact_dir / "export_manifest.json"
        try:
            torch.save(unwrap_model(model).state_dict(), model_only)
            manifest_path.write_text(
                json.dumps(
                    _export_manifest(
                        run_dir=run_dir,
                        config=config,
                        alias=args.checkpoint,
                        checkpoint_path=checkpoint_path,
                        manifest_extra=manifest_extra,
                        input_shape=input_shape,
                        artifact={"weights": str(final_artifact_dir / "model.pt")},
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifact_dir.replace(final_artifact_dir)
        except Exception:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            _cleanup_empty_export_parents(final_artifact_dir, out_dir)
            raise
        print(
            json.dumps(
                {
                    "format": "weights",
                    "weights": str(final_artifact_dir / "model.pt"),
                    "manifest": str(final_artifact_dir / "export_manifest.json"),
                    "decision.threshold": config["decision"]["threshold"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if export_format == "onnx":
        result = _export_onnx(
            model,
            config,
            export_cfg,
            artifact_dir,
            args.checkpoint,
            checkpoint_path,
            manifest_extra,
            run_dir=run_dir,
            input_shape=input_shape,
            published_dir=final_artifact_dir,
        )
        if result != 0:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            _cleanup_empty_export_parents(final_artifact_dir, out_dir)
            return result
        try:
            artifact_dir.replace(final_artifact_dir)
        except Exception:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            _cleanup_empty_export_parents(final_artifact_dir, out_dir)
            raise
        return 0
    shutil.rmtree(artifact_dir, ignore_errors=True)
    _cleanup_empty_export_parents(final_artifact_dir, out_dir)
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
    *,
    run_dir: Path,
    input_shape: tuple[int, int, int, int, int],
    published_dir: Path,
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
    sample = torch.randn(*input_shape)
    onnx_path = out_dir / "model.onnx"
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
    if verify_samples < 1:
        raise ValueError("export.verify_samples must be >= 1")
    max_diff = 0.0
    # Verify on a batch > 1 so the dynamic batch axis is exercised too.
    verify_shape = (2, *input_shape[1:])
    with torch.no_grad():
        for _ in range(verify_samples):
            batch = torch.randn(*verify_shape)
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
    manifest = _export_manifest(
        run_dir=run_dir,
        config=config,
        alias=alias,
        checkpoint_path=checkpoint_path,
        manifest_extra=manifest_extra,
        input_shape=input_shape,
        artifact={
            "onnx": str(published_dir / "model.onnx"),
            "verification_max_abs_diff": max_diff,
        },
    )
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "format": "onnx",
                "onnx": str(published_dir / "model.onnx"),
                "verification_max_abs_diff": max_diff,
                "decision.threshold": config["decision"]["threshold"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
