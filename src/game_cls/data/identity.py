"""Contract-5 runtime identity for exact dataset resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from game_cls.contract import CONTRACT_VERSION


def _sha256(path: str | Path | None) -> str:
    if path is None or not Path(path).is_file():
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_identity(
    config: dict[str, Any], *, verify_content: bool = True
) -> dict[str, Any]:
    """Compute and optionally verify all data consumed by a training run."""
    data = config.get("data") or {}
    if data.get("synthetic", False):
        return {"contract_version": CONTRACT_VERSION, "mode": "synthetic"}
    if not data.get("train_index"):
        # Small checkpoint unit tests and programmatic model-only use do not
        # necessarily configure a dataset. Production finalized configs do.
        return {"contract_version": CONTRACT_VERSION, "mode": "unconfigured"}

    from game_cls.data.indexing import (
        verify_frame_content_integrity,
        verify_split_bundle,
    )
    from game_cls.data.packed_backend import (
        verify_packed_provenance,
        verify_packed_shards,
        verify_source_bundle_artifacts,
    )

    backend = str(data.get("backend", "png"))
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        frame_index = data.get(f"{split}_index")
        video_index = data.get(f"{split}_video_index")
        if not frame_index and not video_index:
            continue
        if not frame_index or not video_index:
            raise ValueError(
                f"Exact dataset identity requires data.{split}_index and "
                f"data.{split}_video_index together."
            )
        source = verify_source_bundle_artifacts(frame_index, video_index)
        verify_split_bundle(Path(frame_index).parent)
        if backend == "png" and verify_content:
            verify_frame_content_integrity(frame_index)
        record: dict[str, Any] = {
            "bundle_id": source["bundle_id"],
            "bundle_manifest_sha256": source["bundle_manifest_sha256"],
            "frame_index_sha256": _sha256(frame_index),
            "video_index_sha256": _sha256(video_index),
        }
        if backend == "packed_uint8":
            packed_index = data.get(f"{split}_packed_index")
            packed_video_index = data.get(f"{split}_packed_video_index")
            if not packed_index or not packed_video_index:
                raise ValueError(
                    f"Packed exact identity requires data.{split}_packed_index "
                    f"and data.{split}_packed_video_index."
                )
            packed_manifest = Path(packed_index).with_name("packed_manifest.json")
            audit_path = Path(frame_index).with_name("audit.json")
            split_manifest = Path(
                (data.get("split") or {}).get("manifest", "split_manifest.parquet")
            )
            if not split_manifest.is_absolute():
                split_manifest = Path(frame_index).parent / split_manifest.name
            verify_packed_provenance(
                packed_manifest,
                frame_index,
                current_video_index=video_index,
                audit_path=audit_path,
                split_manifest_path=split_manifest,
            )
            if verify_content:
                verify_packed_shards(packed_manifest)
            record.update(
                {
                    "packed_manifest_sha256": _sha256(packed_manifest),
                    "packed_index_sha256": _sha256(packed_index),
                    "packed_video_index_sha256": _sha256(packed_video_index),
                }
            )
        splits[split] = record
    return {
        "contract_version": CONTRACT_VERSION,
        "mode": backend,
        "splits": splits,
        "metadata_sidecar_sha256": _sha256(data.get("metadata_sidecar")),
    }


def dataset_identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
