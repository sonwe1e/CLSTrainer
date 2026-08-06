"""Optional per-video metadata sidecar (step5 P2).

The sidecar is an *additive* source of truth keyed by ``source_video_uid``
(``game::video_id``). It is joined to the video index **after** the split is
derived, so it can never leak frames across splits or drift the split
manifest, and it is not part of any dedup identity. When ``data.metadata_sidecar``
is unset, no sidecar is loaded and every video keeps the legacy
``negative_subtype=None`` / ``sample_weight=1.0`` defaults — training is
byte-identical to before.

Schema (parquet columns):

    source_video_uid   string (primary key)
    negative_subtype   string|null
    scene_type         string|null
    capture_domain     string|null
    difficulty         string|null
    sample_weight      float (default 1.0)
    metadata_version   int
    metadata_fingerprint  string (SHA-256 over the rows)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .video_index import VideoEntry

METADATA_SCHEMA_VERSION = 1


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
    for row in sorted(rows, key=lambda item: str(item["source_video_uid"])):
        digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def read_metadata_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the sidecar into ``{source_video_uid: meta}``.

    Missing or empty sidecar files return an empty mapping (never an error),
    so a config that points at a not-yet-created sidecar behaves like the
    pre-sidecar default.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    _, pq = _pyarrow()
    schema_names = set(pq.read_schema(path).names)
    required = {"source_video_uid"}
    missing = required - schema_names
    if missing:
        raise ValueError(
            f"Metadata sidecar {path} is missing required columns: {sorted(missing)}"
        )
    table = pq.read_table(path, memory_map=False)
    result: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        uid = str(row["source_video_uid"])
        result[uid] = {
            "negative_subtype": row.get("negative_subtype"),
            "scene_type": row.get("scene_type"),
            "capture_domain": row.get("capture_domain"),
            "difficulty": row.get("difficulty"),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "metadata_version": int(row.get("metadata_version", 1)),
        }
    return result


def validate_sidecar_against_index(
    sidecar: dict[str, dict[str, Any]],
    entries: list[VideoEntry],
) -> None:
    """Every sidecar uid must exist in the index; raise otherwise."""
    known = {entry.source_video_uid for entry in entries}
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
        meta = sidecar.get(entry.source_video_uid)
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
    *,
    version: int = METADATA_SCHEMA_VERSION,
) -> str:
    """Write the sidecar atomically and return its fingerprint."""
    pa, pq = _pyarrow()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "source_video_uid": str(row["source_video_uid"]),
                "negative_subtype": row.get("negative_subtype"),
                "scene_type": row.get("scene_type"),
                "capture_domain": row.get("capture_domain"),
                "difficulty": row.get("difficulty"),
                "sample_weight": float(row.get("sample_weight", 1.0)),
                "metadata_version": int(version),
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
        pa.Table.from_pylist(normalized),
        temp,
        compression="zstd",
    )
    temp.replace(path)
    return fingerprint
