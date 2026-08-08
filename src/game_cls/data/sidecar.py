"""Optional per-video metadata sidecar (step5 P2).

The sidecar is an *additive* source of truth keyed by ``stable_source_id``
(``game::video_id``). It is joined to the video index **after** the split is
derived, so it can never leak frames across splits or drift the split
manifest, and it is not part of any dedup identity. When ``data.metadata_sidecar``
is unset, no sidecar is loaded and every video uses
``negative_subtype=None`` / ``sample_weight=1.0`` defaults.

That "missing file reads as no metadata" default is a trap for
``data.hard_negative.enabled``, so ``check_hard_negative_readiness`` refuses
the degenerate combination (enabled + missing/unusable sidecar) at
``config validate`` time and at the training launch.

Schema (parquet columns):

    stable_source_id   string (primary key)
    negative_subtype   string|null
    scene_type         string|null
    capture_domain     string|null
    difficulty         string|null
    sample_weight      float (default 1.0)
    metadata_fingerprint  string (SHA-256 over the rows)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from game_cls.contract import require_parquet_contract, stamp_parquet_table

from .video_index import VideoEntry


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading/writing the metadata sidecar requires pyarrow: "
            "python -m pip install pyarrow"
        ) from exc
    return pa, pq


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["stable_source_id"])):
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def read_metadata_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and authenticate a contract-5 sidecar by stable source id.

    Missing or empty sidecar files return an empty mapping (never an error),
    so a config that points at a not-yet-created sidecar behaves like the
    pre-sidecar default.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    require_parquet_contract(path)
    _, pq = _pyarrow()
    # Use open() so Python's own reference counting closes the OS handle
    # when the with-block exits.  Passing a Path to pq.read_schema /
    # pq.read_table keeps a C++ NativeFile alive until GC fires, which on
    # Windows blocks TemporaryDirectory cleanup and any atomic replacement
    # of the sidecar file.
    with open(path, "rb") as fh:
        schema_names = set(pq.read_schema(fh).names)
    required = {
        "stable_source_id",
        "metadata_fingerprint",
    }
    missing = required - schema_names
    if missing:
        raise ValueError(
            f"Metadata sidecar {path} is missing required columns: {sorted(missing)}"
        )
    with open(path, "rb") as fh:
        rows = pq.read_table(fh).to_pylist()
    if not rows:
        return {}
    recorded_fingerprints = {str(row.get("metadata_fingerprint") or "") for row in rows}
    if len(recorded_fingerprints) != 1 or "" in recorded_fingerprints:
        raise ValueError(f"Metadata sidecar {path} has missing or mixed fingerprints.")
    normalized_for_hash = [
        {key: value for key, value in row.items() if key != "metadata_fingerprint"}
        for row in rows
    ]
    actual_fingerprint = _fingerprint(normalized_for_hash)
    recorded_fingerprint = next(iter(recorded_fingerprints))
    if actual_fingerprint != recorded_fingerprint:
        raise ValueError(
            f"Metadata sidecar {path} fingerprint mismatch: recorded "
            f"{recorded_fingerprint}, actual {actual_fingerprint}."
        )
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        stable_id = str(row["stable_source_id"])
        if stable_id in result:
            raise ValueError(
                f"Metadata sidecar {path} contains duplicate primary key "
                f"stable_source_id={stable_id!r}."
            )
        sample_weight = float(row.get("sample_weight", 1.0))
        if not math.isfinite(sample_weight) or sample_weight <= 0:
            raise ValueError(
                f"Metadata sidecar {path} has invalid sample_weight="
                f"{sample_weight!r} for {stable_id}; weights must be finite "
                "and positive."
            )
        result[stable_id] = {
            "negative_subtype": row.get("negative_subtype"),
            "scene_type": row.get("scene_type"),
            "capture_domain": row.get("capture_domain"),
            "difficulty": row.get("difficulty"),
            "sample_weight": sample_weight,
        }
    return result


def check_hard_negative_readiness(config: dict[str, Any]) -> list[str]:
    """Return the reasons ``data.hard_negative`` would silently degrade.

    ``read_metadata_sidecar`` treats a missing file as "no metadata", which is
    the right default for an optional sidecar but is a trap for
    ``hard_negative.enabled``: every negative would fall into the ordinary
    bucket and training would quietly run plain negative sampling. The
    structural half of the contract (sidecar configured, hard_subtypes
    non-empty, positive mix weights) lives in ``config_schema
    .semantic_validate`` because that layer must stay filesystem-free; this
    function is the filesystem half and is called from
    ``cls-trainer config validate`` and from the training launch.

    Returns an empty list when hard-negative mixing is disabled or ready.
    """
    data_cfg = config.get("data") or {}
    hard_negative = data_cfg.get("hard_negative") or {}
    if not hard_negative.get("enabled", False):
        return []
    problems: list[str] = []
    sidecar_path = data_cfg.get("metadata_sidecar")
    if not sidecar_path:
        # semantic_validate already reports this; keep the list non-empty so a
        # caller that only runs this check still refuses the config.
        return [
            "data.hard_negative.enabled requires data.metadata_sidecar to "
            "point at a per-video metadata parquet."
        ]
    path = Path(sidecar_path)
    if not path.is_file():
        return [
            f"data.metadata_sidecar={sidecar_path} does not exist, but "
            "data.hard_negative.enabled=true. A missing sidecar reads as "
            "'no metadata', so every negative would fall into the ordinary "
            "bucket and training would silently degrade to plain negative "
            "sampling."
        ]
    hard_subtypes = set(hard_negative.get("hard_subtypes") or [])
    if not hard_subtypes:
        return [
            "data.hard_negative.enabled requires a non-empty "
            "data.hard_negative.hard_subtypes."
        ]
    subtype_field = str(hard_negative.get("subtype_field", "negative_subtype"))
    # apply_sidecar only ever joins negative_subtype/sample_weight onto a
    # VideoEntry, so any other field would read as None for every video.
    if subtype_field != "negative_subtype":
        return [
            f"data.hard_negative.subtype_field={subtype_field!r} is never "
            "populated: the sidecar join only sets negative_subtype, so every "
            "video would read None and the hard bucket would stay empty."
        ]
    try:
        sidecar = read_metadata_sidecar(path)
    except (RuntimeError, ValueError) as exc:
        # pyarrow missing: say so instead of crashing a config check.
        return [f"Cannot verify data.metadata_sidecar={sidecar_path}: {exc}"]
    tagged = sorted(
        {
            str(meta.get("negative_subtype"))
            for meta in sidecar.values()
            if meta.get("negative_subtype") in hard_subtypes
        }
    )
    if not tagged:
        present = sorted(
            {
                str(meta.get("negative_subtype"))
                for meta in sidecar.values()
                if meta.get("negative_subtype")
            }
        )
        problems.append(
            f"data.metadata_sidecar={sidecar_path} carries no video whose "
            f"negative_subtype is in data.hard_negative.hard_subtypes="
            f"{sorted(hard_subtypes)} (sidecar rows: {len(sidecar)}; subtypes "
            f"present: {present or 'none'}). The hard bucket would be empty "
            "and sampling would degrade to ordinary negatives."
        )
    return problems


def validate_sidecar_against_index(
    sidecar: dict[str, dict[str, Any]],
    entries: list[VideoEntry],
) -> None:
    """Every sidecar uid must exist in the index; raise otherwise."""
    known = {entry.stable_source_id for entry in entries}
    unknown = [uid for uid in sidecar if uid not in known]
    if unknown:
        raise ValueError(
            "Metadata sidecar references source videos absent from the "
            f"index: {sorted(unknown)[:20]}{'...' if len(unknown) > 20 else ''}"
        )


def apply_sidecar(
    entries: list[VideoEntry],
    sidecar: dict[str, dict[str, Any]],
) -> list[VideoEntry]:
    """Join the sidecar onto the entries (post-split, additive).

    Entries without a sidecar row keep their defaults; nothing is removed.
    """
    if not sidecar:
        return entries
    updated: list[VideoEntry] = []
    for entry in entries:
        meta = sidecar.get(entry.stable_source_id)
        if meta is None:
            updated.append(entry)
            continue
        updated.append(
            replace(
                entry,
                negative_subtype=(
                    meta.get("negative_subtype") or entry.negative_subtype
                ),
                sample_weight=float(meta.get("sample_weight", 1.0)),
            )
        )
    return updated


def write_metadata_sidecar(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> str:
    """Write the sidecar atomically and return its fingerprint."""
    pa, pq = _pyarrow()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        stable_id = str(row.get("stable_source_id") or "")
        if not stable_id:
            raise ValueError("Metadata rows require a non-empty stable_source_id.")
        if stable_id in seen:
            raise ValueError(
                f"Duplicate metadata primary key stable_source_id={stable_id!r}."
            )
        seen.add(stable_id)
        sample_weight = float(row.get("sample_weight", 1.0))
        if not math.isfinite(sample_weight) or sample_weight <= 0:
            raise ValueError(
                f"sample_weight for {stable_id} must be finite and positive, "
                f"got {sample_weight!r}."
            )
        normalized.append(
            {
                "stable_source_id": stable_id,
                "negative_subtype": row.get("negative_subtype"),
                "scene_type": row.get("scene_type"),
                "capture_domain": row.get("capture_domain"),
                "difficulty": row.get("difficulty"),
                "sample_weight": sample_weight,
                "metadata_fingerprint": "",
            }
        )
    fingerprint = _fingerprint(
        [
            {k: v for k, v in item.items() if k != "metadata_fingerprint"}
            for item in normalized
        ]
    )
    for item in normalized:
        item["metadata_fingerprint"] = fingerprint
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp{path.suffix}")
    pq.write_table(
        stamp_parquet_table(pa.Table.from_pylist(normalized)),
        temp,
        compression="zstd",
    )
    temp.replace(path)
    return fingerprint
