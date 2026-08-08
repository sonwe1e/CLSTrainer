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

``benchmark.gate_metrics`` uses explicit metric names and comparison
operators::

    benchmark:
      gate_metrics:
        global_fpr_at_decision_threshold: {op: "<=", value: 0.01}
        global_positive_recall_at_decision_threshold: {op: ">=", value: 0.80}

A bare scalar bound (``global_fpr_at_decision_threshold: 0.01``) stays legal
and means "the natural bound for this metric", resolved through the
``_LOWER_BETTER`` / ``_HIGHER_BETTER`` tables below. Metric names are checked
against what ``engine.evaluator.evaluate`` actually emits, so a typo or a
metric with no documented direction is a config error rather than a gate that
silently reads ``metric absent`` after a full benchmark run.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MINING_MANIFEST_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes; empty string when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_spec_fingerprint(gate_metrics: dict[str, Any] | None) -> str:
    """Canonical SHA-256 of the gate contract (sorted, stable JSON)."""
    return hashlib.sha256(
        json.dumps(gate_metrics or {}, sort_keys=True).encode("utf-8")
    ).hexdigest()


def canonical_config_sha256(config: dict[str, Any] | None) -> str:
    """Canonical SHA-256 of a *finalized* config's content (audit P0-2/P0-3).

    Hashing the config object rather than the file it came from is what makes
    the provenance honest: ``benchmark evaluate --config OTHER.yaml`` evaluates
    OTHER.yaml, so recording ``file_sha256(run_dir/"resolved_config.json")``
    would describe a config the run never used (audit P0-3).

    Both the writer (``benchmark evaluate``) and the reader (``release
    check``) MUST route through this one function. Two call sites that
    canonicalize even slightly differently -- a different ``default=``, an
    unsorted dump -- produce different digests for identical configs, which
    would turn the P0-2 identity comparison into an unconditional failure.
    """
    return hashlib.sha256(
        json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

# ---------------------------------------------------------------------------
# Gate contract (step6): explicit metric name + comparison operator.
# ---------------------------------------------------------------------------
#
# A gate is ``{metric_name: {op, value}}``. Bare scalars stay legal for
# backward compatibility, but the comparison direction is then looked up in an
# explicit table instead of guessed from substrings of the metric name -- the
# old ``"fpr" in name`` heuristic mishandled specificity (upper-bounded a
# higher-is-better metric), ECE and the negative-score percentiles
# (lower-bounded lower-is-better metrics).

GATE_OPERATORS: tuple[str, ...] = ("<=", "<", ">=", ">")

# Lower is better: a bare scalar bound is an UPPER bound (op "<=").
_LOWER_BETTER: frozenset[str] = frozenset(
    {
        # Mistake counts.
        "fp",
        "fn",
        # False-positive rates.
        "global_fpr_at_decision_threshold",
        "worst_game_fpr_at_decision_threshold",
        "worst_subtype_fpr_at_decision_threshold",
        # Losses.
        "cross_entropy",
        "threshold_loss",
        "objective_loss",
        # Calibration error.
        "brier_score",
        "ece_20_bins",
        "ece_tail_95_100",
        # Negative-score tail: a negative pair should score low.
        "negative_score_p99",
        "negative_score_p999",
        "negative_score_max",
        "negative_margin_p10",
        "negative_margin_p50",
        "negative_margin_p90",
    }
)

# Higher is better: a bare scalar bound is a LOWER bound (op ">=").
_HIGHER_BETTER: frozenset[str] = frozenset(
    {
        # Correct counts.
        "tp",
        "tn",
        # Plain binary metrics (BinaryMetrics.to_dict()).
        "precision",
        "recall",
        "f1",
        "accuracy",
        "specificity",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        # F1 family, neutral names plus the legacy _tau099 aliases.
        "global_f1_at_decision_threshold",
        "macro_game_f1_at_decision_threshold",
        "worst_game_f1_at_decision_threshold",
        "global_f1_tau099",
        "macro_game_f1_tau099",
        "worst_game_f1_tau099",
        # Specificity / recall at the decision threshold.
        "global_specificity_at_decision_threshold",
        "global_positive_recall_at_decision_threshold",
        "worst_game_positive_recall_at_decision_threshold",
        "worst_subtype_recall_at_decision_threshold",
        # Margin pass rates and the positive-margin quantiles.
        "positive_margin_pass_rate",
        "negative_margin_pass_rate",
        "positive_margin_p10",
        "positive_margin_p50",
        "positive_margin_p90",
        # Low-FPR protocol.
        "recall_at_max_fpr",
        "low_fpr_partial_auc",
    }
)

# Numeric metrics the evaluator emits that describe the *run* rather than its
# quality, so "better" has no direction. Gating them is legal but requires the
# explicit ``{op, value}`` form (e.g. ``sample_count: {op: ">=", value: 10000}``
# to refuse a release measured on a truncated challenge set).
_NON_DIRECTIONAL: frozenset[str] = frozenset(
    {
        "sample_count",
        "threshold",
        "threshold_loss_weight",
    }
)


def gateable_metric_names() -> frozenset[str]:
    """Every metric name a gate may reference.

    This is exactly the set of numeric scalars ``engine.evaluator.evaluate``
    puts in its metrics dict; ``tests/test_benchmark_scan.py`` pins the two
    sets together so a new evaluator metric cannot silently become
    un-gateable (or a gate name silently become unproducible).
    """
    return _LOWER_BETTER | _HIGHER_BETTER | _NON_DIRECTIONAL


def _legacy_alias_pairs() -> dict[str, str]:
    """Bidirectional canonical<->legacy metric-name map.

    Reuses ``selection._LEGACY_METRIC_ALIASES`` instead of copying it so the
    ``*_at_decision_threshold`` / ``*_tau099`` contract keeps exactly one
    source of truth.
    """
    from game_cls.engine.training.selection import _LEGACY_METRIC_ALIASES

    pairs = dict(_LEGACY_METRIC_ALIASES)
    pairs.update({legacy: canonical for canonical, legacy in pairs.items()})
    return pairs


def _gate_metric_value(metrics: dict, name: str) -> Any:
    """Read a metric, falling back to its legacy/canonical alias."""
    value = metrics.get(name)
    if value is not None:
        return value
    alias = _legacy_alias_pairs().get(name)
    return metrics.get(alias) if alias is not None else None


def _resolve_gate(name: str, spec: Any) -> tuple[str, float]:
    """Normalize one gate entry into ``(op, bound)``.

    Raises ``ValueError`` for an unknown metric name, an unknown operator, a
    malformed spec, or a bare scalar on a metric with no documented
    direction. ``validate_gate_metrics`` reports the same conditions as
    config problems; this function is the runtime backstop for a gate dict
    that never went through config validation.
    """
    if isinstance(spec, dict):
        unknown_fields = sorted(set(spec) - {"op", "value"})
        if unknown_fields:
            raise ValueError(
                f"benchmark.gate_metrics.{name} has unknown field(s) "
                f"{unknown_fields}; a gate is {{op, value}}."
            )
        if "op" not in spec or "value" not in spec:
            raise ValueError(
                f"benchmark.gate_metrics.{name} must set both 'op' and "
                f"'value' (got {sorted(spec)})."
            )
        op = str(spec["op"])
        if op not in GATE_OPERATORS:
            raise ValueError(
                f"benchmark.gate_metrics.{name}.op={op!r} is not a comparison "
                f"operator; use one of {list(GATE_OPERATORS)}."
            )
        raw_value = spec["value"]
    else:
        # Backward compatibility: a bare scalar means "the natural bound for
        # this metric", which only exists when the direction is documented.
        if name in _LOWER_BETTER:
            op = "<="
        elif name in _HIGHER_BETTER:
            op = ">="
        else:
            raise ValueError(
                f"benchmark.gate_metrics.{name} is a bare scalar bound, but "
                f"{name} has no documented better-direction, so the "
                "comparison would have to be guessed. Use the explicit form: "
                f'{name}: {{op: "<=", value: {spec!r}}}.'
            )
        raw_value = spec
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"benchmark.gate_metrics.{name} bound must be a number (got {raw_value!r})."
        )
    return op, float(raw_value)


def _compare(value: float, op: str, bound: float) -> bool:
    if op == "<=":
        return value <= bound
    if op == "<":
        return value < bound
    if op == ">=":
        return value >= bound
    if op == ">":
        return value > bound
    raise ValueError(f"Unsupported gate operator: {op!r}")


def validate_gate_metrics(gate_metrics: Any) -> list[str]:
    """Return a list of config problems in ``benchmark.gate_metrics``.

    Checks the metric names against what the evaluator can actually produce
    and the operators against ``GATE_OPERATORS``, so a typo fails at config
    time instead of showing up as ``metric absent`` after a full benchmark
    run.
    """
    problems: list[str] = []
    if gate_metrics in (None, {}):
        return problems
    if not isinstance(gate_metrics, dict):
        return ["benchmark.gate_metrics must be a mapping of metric name to bound."]
    known = gateable_metric_names()
    for name, spec in gate_metrics.items():
        metric_name = str(name)
        if metric_name not in known:
            suggestion = difflib.get_close_matches(
                metric_name, sorted(known), n=1, cutoff=0.5
            )
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            problems.append(
                f"benchmark.gate_metrics.{metric_name} is not a metric the "
                f"evaluator produces, so the gate could never pass.{hint} "
                "See 'cls-trainer config reference' for benchmark.gate_metrics."
            )
            continue
        try:
            _resolve_gate(metric_name, spec)
        except ValueError as exc:
            problems.append(str(exc))
    return problems


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
    # Write to a temp sibling then atomically replace, which also releases
    # the file handle on Windows (avoids PermissionError on temp-dir cleanup).
    temp = path.with_suffix(f".tmp{path.suffix}")
    pq.write_table(pa.Table.from_pylist(normalized), temp, compression="zstd")
    temp.replace(path)


def read_mining_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a mining manifest, rejecting unknown format versions."""
    _, pq = _pyarrow()
    # Pass an already-opened file object so Python's own reference
    # counting releases the OS handle when the with-block exits.  Passing
    # a Path to pq.read_table keeps a C++ NativeFile alive until GC
    # fires, which on Windows blocks TemporaryDirectory cleanup and any
    # subsequent rename of the same file (re-annotation, atomic replace).
    with open(path, "rb") as fh:
        rows = pq.read_table(fh).to_pylist()
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
                # Audit P0-5: consume the persisted canonical uid from the batch
                # meta instead of re-deriving ``game::video_id``; the legacy
                # fallback only covers frame-based PairSample metas that never
                # carried one.
                uid = meta.get("source_video_uid") or (
                    f"{meta['game']}::{meta['video_id']}"
                )
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


def _scores_from_metrics(
    metrics: dict, gate_metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize a challenge evaluation into a benchmark report payload.

    Every gated metric is included on top of the fixed summary, so a report
    always carries the numbers its own gates were judged on and
    ``config validate --release`` can re-check them.
    """
    scores = {
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
    for name in gate_metrics or {}:
        scores.setdefault(str(name), _gate_metric_value(metrics, str(name)))
    return scores


def check_gates(
    metrics: dict, gate_metrics: dict[str, Any]
) -> list[tuple[str, bool, str]]:
    """Return ``(metric_name, passed, detail)`` for each configured gate.

    Each gate is either the explicit ``{op, value}`` form or a bare scalar
    whose direction comes from the ``_LOWER_BETTER`` / ``_HIGHER_BETTER``
    tables. A malformed gate raises ``ValueError``: the gate dict is
    validated at config load (``semantic_validate``), so reaching this point
    with a bad op or an unknown metric name means the dict bypassed
    validation and must not be reported as a silent pass.
    """
    checks: list[tuple[str, bool, str]] = []
    for raw_name, spec in (gate_metrics or {}).items():
        name = str(raw_name)
        if name not in gateable_metric_names():
            raise ValueError(
                f"benchmark.gate_metrics.{name} is not a metric the evaluator "
                "produces; the gate could never pass."
            )
        op, bound = _resolve_gate(name, spec)
        value = _gate_metric_value(metrics, name)
        if value is None:
            # The name is producible in principle, so this run really did not
            # produce it (e.g. worst_subtype_fpr without subtype grouping).
            checks.append(
                (name, False, f"metric absent (required {op} {bound:.4g})"),
            )
            continue
        passed = _compare(float(value), op, bound)
        checks.append((name, passed, f"value={float(value):.4g} {op} {bound:.4g}"))
    return checks


def write_benchmark_report(
    output_dir: Path,
    *,
    run_id: str,
    checkpoint_alias: str,
    metrics: dict,
    gates: list[tuple[str, bool, str]],
    grouped_metrics: dict | None,
    gate_metrics: dict[str, Any] | None = None,
    checkpoint_sha256: str = "",
    resolved_config_sha256: str = "",
    challenge_dataset_fingerprint: str = "",
    gate_spec_fingerprint: str = "",
) -> Path:
    """Write ``benchmarks/<run>_<alias>/report.json`` and return its path.

    ``gate_metrics`` is the raw configured gate dict; recording it makes the
    report self-describing, so a release check can tell a gate that really
    passed from a report measured under a different contract. The identity
    fields (audit P0-4) bind the PASS to the exact artifact it was earned on:
    ``checkpoint_sha256`` is the released model's hash, and the config /
    challenge-dataset / gate-spec fingerprints pin the other three legs of the
    release identity tuple.
    """
    report_dir = output_dir / f"{run_id or 'run'}_{checkpoint_alias}"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "checkpoint": checkpoint_alias,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "challenge_dataset_fingerprint": challenge_dataset_fingerprint,
        "gate_spec_fingerprint": gate_spec_fingerprint,
        "scores": _scores_from_metrics(metrics, gate_metrics),
        "gate_metrics": gate_metrics or {},
        "gates": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in gates
        ],
        "grouped": grouped_metrics or {},
    }
    path = report_dir / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
