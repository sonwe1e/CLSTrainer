"""Hard-negative mining and fixed challenge-set benchmarking (step5 P3/P6).

``scan_negative_pool`` runs a best checkpoint over a training-side negative
pool, ranks negatives by ``p_positive``, keeps a bounded top-K per source
video (so continuous frames cannot drown the manifest), deduplicates by pair
identity, and writes a versioned ``hard_negatives.parquet`` manifest.

``evaluate_challenge`` runs a best checkpoint over a fixed, immutable
challenge set and reports global/worst-game/worst-subtype FPR, recall and
negative-score percentiles, plus the configured ``benchmark.gate_metrics``
gates. The challenge set is *never* part of the train/val/test protocol and
never feeds model selection.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MINING_MANIFEST_VERSION = 1


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Benchmark parquet I/O requires pyarrow: python -m pip install pyarrow"
        ) from exc
    return pa, pq


def write_mining_manifest(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write the versioned hard-negative mining manifest."""
    pa, pq = _pyarrow()
    normalized = []
    for row in rows:
        normalized.append(
            {
                "source_video_uid": str(row["source_video_uid"]),
                "game": row.get("game", ""),
                "video_id": row.get("video_id", ""),
                "frame0_id": int(row.get("frame0_id", 0)),
                "frame1_id": int(row.get("frame1_id", 0)),
                "delta": int(row.get("delta", 2)),
                "p_positive": float(row.get("p_positive", 0.0)),
                "subtype_before": row.get("subtype_before"),
                "rank_in_video": int(row.get("rank_in_video", 0)),
                "mining_version": int(MINING_MANIFEST_VERSION),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(normalized), path, compression="zstd")


def read_mining_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a mining manifest, rejecting unknown format versions."""
    _, pq = _pyarrow()
    # memory_map=False avoids a lingering Windows file handle that would
    # make TemporaryDirectory cleanup (and re-annotation) fail.
    table = pq.read_table(Path(path), memory_map=False)
    rows = table.to_pylist()
    for row in rows:
        version = int(row.get("mining_version", 0))
        if version != MINING_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported mining manifest version {version} "
                f"(expected {MINING_MANIFEST_VERSION})."
            )
    return rows


def scan_negative_pool(
    model,
    loader,
    device,
    *,
    top_k_per_video: int,
    score_threshold: float | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Score every negative pair in ``loader`` and build the mining rows.

    ``loader`` must yield batches with ``images`` (uint8 [B,2,3,H,W]),
    ``labels`` and ``meta`` (per-sample game/video_id/frame0_id/frame1_id/
    delta). Only label-0 (negative) pairs are kept.
    """
    import torch

    from game_cls.engine.device import autocast_context

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            if images.dtype == torch.uint8:
                images = images.float().div_(255.0)
            labels = batch["labels"].to(device)
            with autocast_context(device, False, "bfloat16"):
                logits = model(images[:, 0], images[:, 1])
            probability = torch.softmax(logits, dim=1)[:, 1]
            for index in range(len(labels)):
                if int(labels[index]) != 0:
                    continue
                meta = batch["meta"][index]
                uid = f"{meta['game']}::{meta['video_id']}"
                p_positive = float(probability[index].item())
                if score_threshold is not None and p_positive < score_threshold:
                    continue
                by_video[uid].append(
                    {
                        "source_video_uid": uid,
                        "game": meta["game"],
                        "video_id": meta["video_id"],
                        "frame0_id": int(meta["frame0_id"]),
                        "frame1_id": int(meta["frame1_id"]),
                        "delta": int(meta["delta"]),
                        "p_positive": p_positive,
                        "subtype_before": meta.get("negative_subtype"),
                    }
                )
    rows: list[dict[str, Any]] = []
    for candidates in by_video.values():
        candidates.sort(key=lambda item: item["p_positive"], reverse=True)
        # Dedup by pair identity, then keep top-K.
        seen: set[tuple[int, int, int]] = set()
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            identity = (
                candidate["frame0_id"],
                candidate["frame1_id"],
                candidate["delta"],
            )
            if identity in seen:
                continue
            seen.add(identity)
            kept.append(candidate)
            if len(kept) >= top_k_per_video:
                break
        for rank, candidate in enumerate(kept):
            candidate["rank_in_video"] = rank
            rows.append(candidate)
    rows.sort(key=lambda item: item["p_positive"], reverse=True)
    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]
    return rows


def _scores_from_metrics(metrics: dict) -> dict[str, Any]:
    """Summarize a challenge evaluation into a benchmark report payload."""
    return {
        "sample_count": metrics.get("sample_count"),
        "global_fpr_at_decision_threshold": metrics.get(
            "global_fpr_at_decision_threshold"
        ),
        "global_positive_recall_at_decision_threshold": metrics.get(
            "global_positive_recall_at_decision_threshold"
        ),
        "worst_game_fpr_at_decision_threshold": metrics.get(
            "worst_game_fpr_at_decision_threshold"
        ),
        "worst_game_positive_recall_at_decision_threshold": metrics.get(
            "worst_game_positive_recall_at_decision_threshold"
        ),
        "worst_subtype_fpr_at_decision_threshold": metrics.get(
            "worst_subtype_fpr_at_decision_threshold"
        ),
        "negative_score_p99": metrics.get("negative_score_p99"),
        "negative_score_p999": metrics.get("negative_score_p999"),
        "selection_eligible": metrics.get("selection_eligible"),
    }


def check_gates(
    metrics: dict, gate_metrics: dict[str, Any]
) -> list[tuple[str, bool, str]]:
    """Return ``(metric_name, passed, detail)`` for each configured gate."""
    checks: list[tuple[str, bool, str]] = []
    for name, bound in (gate_metrics or {}).items():
        value = metrics.get(name)
        if value is None:
            checks.append((name, False, "metric absent"))
            continue
        # FPR/recall-style gates: value must be <= bound (FPR) or >= bound
        # (recall). Bound direction is inferred from the metric name.
        if "fpr" in name.lower() or "specificity" in name.lower():
            passed = float(value) <= float(bound)
        else:
            passed = float(value) >= float(bound)
        checks.append(
            (name, passed, f"value={float(value):.4g} bound={float(bound):.4g}")
        )
    return checks


def write_benchmark_report(
    output_dir: Path,
    *,
    run_id: str,
    checkpoint_alias: str,
    metrics: dict,
    gates: list[tuple[str, bool, str]],
    grouped_metrics: dict | None,
) -> Path:
    """Write ``benchmarks/<run>_<alias>/report.json`` and return its path."""
    report_dir = output_dir / f"{run_id or 'run'}_{checkpoint_alias}"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "checkpoint": checkpoint_alias,
        "scores": _scores_from_metrics(metrics),
        "gates": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in gates
        ],
        "grouped": grouped_metrics or {},
    }
    path = report_dir / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
