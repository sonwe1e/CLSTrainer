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

import datetime
import difflib
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from game_cls.contract import (
    CONTRACT_VERSION,
    require_parquet_contract,
    stamp_parquet_table,
)


def benchmark_source_fingerprint() -> str:
    """Hash the source modules that define benchmark/evaluator semantics."""
    package_root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "data/collate.py",
        "data/lazy_pair_dataset.py",
        "engine/evaluator.py",
        "engine/training/selection.py",
        "metrics/binary_metrics.py",
        "reports/benchmark.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(
                f"Cannot fingerprint benchmark contract; source is missing: {path}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def benchmark_output_dir(config: dict[str, Any], run_dir: str | Path) -> Path:
    """Resolve benchmark reports independently of the caller's current CWD."""
    configured = Path((config.get("benchmark") or {}).get("output_dir", "benchmarks"))
    return configured if configured.is_absolute() else Path(run_dir) / configured


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
# Challenge bundle fingerprint (audit P0-4).
# ---------------------------------------------------------------------------

# Component keys that are per-file SHA-256 digests, in the order they are
# reported. Used to decide "did any leg resolve at all" and to diff two
# recorded component sets.
_CHALLENGE_FILE_LEGS: tuple[str, ...] = (
    "challenge_index_sha256",
    "challenge_video_index_sha256",
    "challenge_packed_index_sha256",
    "challenge_packed_video_index_sha256",
    "challenge_metadata_sha256",
    "challenge_bundle_manifest_sha256",
    "packed_manifest_sha256",
)


def challenge_bundle_fingerprint(config: dict[str, Any] | None) -> tuple[str, dict]:
    """Fingerprint the whole challenge bundle, not just its video index (P0-4).

    The previous definition was ``file_sha256(packed_video_index or
    video_index)`` -- a single file. That left the release identity blind to
    changes that provably move the gated numbers: editing the metadata sidecar
    changes ``worst_subtype_fpr`` while the video index stays byte-identical,
    and PNGs can be replaced in place as long as the index is not regenerated.

    Returns ``(digest, components)``. ``components`` is recorded alongside the
    digest so a mismatch can name the leg that moved instead of leaving an
    operator to compare two opaque hashes.

    Paths are read off ``config["data"]`` here rather than passed in, because
    the writer (``benchmark evaluate``) and the reader (``release check``)
    compare these digests for equality. An inlined copy on either side would
    silently drift and turn every comparison into an unconditional failure --
    the same failure mode ``canonical_config_sha256`` documents.

    ``packed_manifest.json`` is hashed rather than the shards themselves: the
    manifest already carries a ``shard_sha256`` map, so every shard's bytes are
    transitively pinned without re-reading gigabytes of shard data.

    Pixel content is covered transitively too. The challenge frame index
    carries a per-frame ``content_sha256``, so hashing the index file detects an
    edited PNG -- but only for a bundle built with content hashing enabled.
    ``components["covers_content"]`` records whether that held rather than
    assuming it, so a bundle indexed without content hashes is visibly weaker
    instead of quietly weaker.

    When no leg resolves at all the digest is ``""``, which keeps the existing
    fail-closed behaviour in ``release check``: an identity this side cannot
    recompute is refused rather than implicitly accepted.
    """
    config = config or {}
    data_cfg = config.get("data") or {}

    def _hash(key: str) -> str:
        path = data_cfg.get(key)
        return file_sha256(path) if path else ""

    packed_index = data_cfg.get("challenge_packed_index")
    packed_manifest = ""
    challenge_index = data_cfg.get("challenge_index")
    challenge_video_index = data_cfg.get("challenge_video_index")
    bundle_manifest = ""
    if challenge_index:
        bundle_manifest = file_sha256(
            Path(challenge_index).with_name("bundle_manifest.json")
        )
        from game_cls.data.indexing import (
            verify_external_bundle,
            verify_frame_content_integrity,
        )

        verify_external_bundle(Path(challenge_index).parent, pool="challenge")
        if not packed_index:
            verify_frame_content_integrity(challenge_index)
    if packed_index:
        packed_manifest_path = Path(packed_index).with_name("packed_manifest.json")
        packed_manifest = file_sha256(packed_manifest_path)
        from game_cls.data.packed_backend import (
            verify_packed_provenance,
            verify_packed_shards,
        )

        verify_packed_provenance(
            packed_manifest_path,
            challenge_index,
            current_video_index=challenge_video_index,
            audit_path=(
                Path(challenge_index).with_name("audit.json")
                if challenge_index
                else None
            ),
        )
        verify_packed_shards(packed_manifest_path)

    components: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "challenge_index_sha256": _hash("challenge_index"),
        "challenge_video_index_sha256": _hash("challenge_video_index"),
        "challenge_packed_index_sha256": _hash("challenge_packed_index"),
        "challenge_packed_video_index_sha256": _hash("challenge_packed_video_index"),
        "challenge_metadata_sha256": _hash("challenge_metadata"),
        "challenge_bundle_manifest_sha256": bundle_manifest,
        "packed_manifest_sha256": packed_manifest,
    }

    # Preprocessing: the same weights on the same frames produce different
    # metrics under a different image geometry, pair delta or decision
    # threshold, so these belong to the challenge identity.
    image_spec: dict[str, Any] = {}
    try:
        from game_cls.data.image_spec import ImageSpec

        spec = ImageSpec.from_config(data_cfg)
        image_spec = {
            "channels": spec.channels,
            "height": spec.height,
            "width": spec.width,
        }
    except (KeyError, ValueError, ImportError):
        # A config without width/height cannot describe a geometry; leaving it
        # empty is honest and still hashes distinctly from a real spec.
        image_spec = {}
    components["image_spec"] = image_spec
    components["preprocessing"] = {
        "test_delta": (config.get("pair") or {}).get("test_delta"),
        "decision_threshold": (config.get("decision") or {}).get("threshold"),
        "group_by_negative_subtype": (config.get("evaluation") or {}).get(
            "group_by_negative_subtype"
        ),
    }

    # Whether the index's own content hashes back the digest. Derived, so it is
    # reported but deliberately NOT hashed: it is a property of the recorded
    # legs, not an independent input.
    components["covers_content"] = _challenge_covers_content(
        data_cfg.get("challenge_index")
    )

    if not any(components[leg] for leg in _CHALLENGE_FILE_LEGS):
        return "", components

    payload = {key: components[key] for key in components if key != "covers_content"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return digest, components


def _challenge_covers_content(challenge_index: str | Path | None) -> bool | None:
    """Whether every frame in the challenge index carries a content hash.

    ``None`` when it cannot be determined (no index configured, absent file, or
    an unreadable/foreign parquet) -- distinct from ``False``, which is a
    positive finding that the index was built without content hashing.
    """
    if not challenge_index or not Path(challenge_index).is_file():
        return None
    try:
        from game_cls.data.indexing import read_frame_parquet
        from game_cls.data.splitter import fingerprint_covers_content

        frames = read_frame_parquet(challenge_index)
    except Exception:
        # Reading the index is a best-effort diagnostic; it must never be the
        # reason a benchmark or a release check crashes.
        return None
    if not frames:
        return None
    return fingerprint_covers_content(frames)


def challenge_component_differences(
    recorded: dict[str, Any] | None, current: dict[str, Any] | None
) -> list[str]:
    """Name the challenge-bundle legs that differ between two component sets.

    Turns "two digests differ" into "challenge_metadata_sha256 differs", which
    is the difference between an operator knowing what to re-run and guessing.
    """
    recorded = recorded or {}
    current = current or {}
    if not recorded:
        return []
    differences: list[str] = []
    keys = [
        "contract_version",
        *_CHALLENGE_FILE_LEGS,
        "image_spec",
        "preprocessing",
    ]
    for key in keys:
        if key not in recorded and key not in current:
            continue
        was, now = recorded.get(key), current.get(key)
        if was != now:
            differences.append(f"{key}: report={was!r} current={now!r}")
    return differences


# ---------------------------------------------------------------------------
# Gate contract (step6): explicit metric name + comparison operator.
# ---------------------------------------------------------------------------
#
# A gate is always ``{metric_name: {op, value}}``.

GATE_OPERATORS: tuple[str, ...] = ("<=", "<", ">=", ">")

# Metrics whose quality direction is lower-is-better.
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

# Metrics whose quality direction is higher-is-better.
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
        # F1 family.
        "global_f1_at_decision_threshold",
        "macro_game_f1_at_decision_threshold",
        "worst_game_f1_at_decision_threshold",
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


def _gate_metric_value(metrics: dict, name: str) -> Any:
    """Read one canonical evaluator metric."""
    return metrics.get(name)


def _resolve_gate(name: str, spec: Any) -> tuple[str, float]:
    """Normalize one gate entry into ``(op, bound)``.

    Raises ``ValueError`` for an unknown metric name, an unknown operator, a
    malformed spec, or a scalar gate. ``validate_gate_metrics`` reports the same conditions as
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
        raise ValueError(f"benchmark.gate_metrics.{name} must use {{op, value}}.")
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
    """Write the contract-5 hard-negative mining manifest."""
    pa, pq = _pyarrow()
    normalized = []
    for row in rows:
        normalized.append(
            {
                "stable_source_id": str(row["stable_source_id"]),
                "source_version_id": str(row["source_version_id"]),
                "game": row.get("game", ""),
                "video_id": row.get("video_id", ""),
                "frame0_id": int(row.get("frame0_id", 0)),
                "frame1_id": int(row.get("frame1_id", 0)),
                "delta": int(row.get("delta", 2)),
                "p_positive": float(row.get("p_positive", 0.0)),
                "subtype_before": row.get("subtype_before"),
                "rank_in_video": int(row.get("rank_in_video", 0)),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp sibling then atomically replace, which also releases
    # the file handle on Windows (avoids PermissionError on temp-dir cleanup).
    temp = path.with_suffix(f".tmp{path.suffix}")
    pq.write_table(
        stamp_parquet_table(pa.Table.from_pylist(normalized)),
        temp,
        compression="zstd",
    )
    temp.replace(path)


def read_mining_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a contract-5 mining manifest."""
    _, pq = _pyarrow()
    require_parquet_contract(path)
    # Pass an already-opened file object so Python's own reference
    # counting releases the OS handle when the with-block exits.  Passing
    # a Path to pq.read_table keeps a C++ NativeFile alive until GC
    # fires, which on Windows blocks TemporaryDirectory cleanup and any
    # subsequent rename of the same file (re-annotation, atomic replace).
    with open(path, "rb") as fh:
        rows = pq.read_table(fh).to_pylist()
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
                # Consume the persisted canonical identity from batch metadata.
                stable_id = str(meta["stable_source_id"])
                version_id = str(meta["source_version_id"])
                p_positive = float(probability[index].item())
                if score_threshold is not None and p_positive < score_threshold:
                    continue
                by_video[version_id].append(
                    {
                        "stable_source_id": stable_id,
                        "source_version_id": version_id,
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
    challenge_bundle_components: dict[str, Any] | None = None,
    benchmark_source_sha256: str = "",
) -> Path:
    """Append one immutable release-identity report and return its path.

    ``gate_metrics`` is the raw configured gate dict; recording it makes the
    report self-describing, so a release check can tell a gate that really
    passed from a report measured under a different contract. The identity
    fields (audit P0-4) bind the PASS to the exact artifact it was earned on:
    ``checkpoint_sha256`` is the released model's hash, and the config /
    challenge-dataset / gate-spec fingerprints pin the other three legs of the
    release identity tuple.

    ``challenge_bundle_components`` is the per-leg breakdown behind
    ``challenge_dataset_fingerprint``. It is recorded so a release check that
    sees a differing digest can name the leg that moved (the sidecar, the frame
    index, the image geometry) rather than reporting two opaque hashes.
    """
    from game_cls.release import release_identity_sha256

    identity = {
        "run_id": str(run_id),
        "checkpoint_sha256": str(checkpoint_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "challenge_dataset_fingerprint": str(challenge_dataset_fingerprint),
        "gate_spec_fingerprint": str(gate_spec_fingerprint),
        "contract_version": CONTRACT_VERSION,
        "benchmark_source_sha256": str(
            benchmark_source_sha256 or benchmark_source_fingerprint()
        ),
    }
    identity_sha = release_identity_sha256(identity)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report_dir = (
        output_dir
        / (run_id or "run")
        / (checkpoint_sha256 or "unbound")
        / identity_sha
        / f"{stamp}-{uuid4().hex[:8]}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "checkpoint": checkpoint_alias,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "challenge_dataset_fingerprint": challenge_dataset_fingerprint,
        "challenge_bundle_components": challenge_bundle_components or {},
        "gate_spec_fingerprint": gate_spec_fingerprint,
        "contract_version": CONTRACT_VERSION,
        "benchmark_source_sha256": (
            benchmark_source_sha256 or benchmark_source_fingerprint()
        ),
        "release_identity_sha256": identity_sha,
        "scores": _scores_from_metrics(metrics, gate_metrics),
        "gate_metrics": gate_metrics or {},
        "gates": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in gates
        ],
        "grouped": grouped_metrics or {},
    }
    path = report_dir / "report.json"
    temp = report_dir / ".report.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return path
