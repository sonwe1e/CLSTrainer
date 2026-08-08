from __future__ import annotations

import contextlib
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from .image_spec import ImageSpec


def _file_sha256(path: str | Path | None) -> str:
    """SHA-256 of a file's bytes; empty string when absent."""
    if path is None:
        return ""
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bundle_id(frame_index: str | Path | None) -> str:
    """``bundle_id`` of the split bundle a frame index belongs to (audit P0-9).

    Read from the ``bundle_manifest.json`` sitting beside the frame index.
    Empty when the index is not part of a sealed bundle, which keeps packing an
    unsealed legacy directory possible -- the binding then simply cannot be
    asserted later, and ``verify_packed_provenance`` says so rather than
    inventing agreement.
    """
    if frame_index is None:
        return ""
    manifest = Path(frame_index).parent / "bundle_manifest.json"
    if not manifest.is_file():
        return ""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("bundle_id") or "")


def verify_packed_provenance(
    manifest_path: str | Path,
    current_frame_index: str | Path | None,
    *,
    audit_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
) -> None:
    """Refuse stale packed data before a DataLoader is built (audit P0-6).

    A packed shard set is bound to the exact frame index, audit and split
    manifest it was generated from. If any of those changed after packing, the
    shards no longer match the current indexes and training must refuse instead
    of silently reading old pixels over new index files.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Packed shard manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Packed manifest is unreadable: {manifest_path}: {exc}") from exc

    def _check(field: str, current_path: str | Path | None) -> None:
        recorded = manifest.get(field)
        if not recorded:
            # A manifest that predates the provenance binding cannot prove the
            # shards match the current indexes; refuse and demand a repack.
            raise RuntimeError(
                f"Packed manifest {manifest_path} has no {field}; it predates "
                "the provenance binding. Re-run 'cls-trainer dataset pack' to "
                "regenerate provenance-bound shards."
            )
        # Audit P0-6: _file_sha256 returns "" for a missing/unreadable file,
        # so the previous ``if current and current != recorded`` guard treated
        # ABSENCE as agreement -- deleting or moving the source index made the
        # staleness check silently pass. An unhashable source cannot prove the
        # shards match, so it is a failure, and the message distinguishes the
        # two causes.
        current = _file_sha256(current_path)
        if not current:
            raise RuntimeError(
                f"Packed provenance cannot be verified: {field} refers to "
                f"{current_path}, which is missing or unreadable. The manifest "
                f"records {recorded[:16]}..., but the current file cannot be "
                "hashed, so staleness cannot be ruled out. Restore the source "
                "or re-run 'cls-trainer dataset pack'."
            )
        if current != recorded:
            raise RuntimeError(
                f"Packed data is stale: {field} changed since packing "
                f"(manifest {recorded[:16]}..., current {current[:16]}...). "
                "The current indexes were regenerated but the shards were not; "
                "re-run 'cls-trainer dataset pack'."
            )

    # The frame index binding is mandatory -- a manifest without it cannot
    # prove anything. The audit/split-manifest bindings are optional: they are
    # only checked when the packer recorded them (the pack may legitimately
    # have had no audit/split-manifest to bind to).
    _check("source_frame_index_sha256", current_frame_index)
    if audit_path is not None and manifest.get("audit_fingerprint"):
        _check("audit_fingerprint", audit_path)
    if split_manifest_path is not None and manifest.get(
        "split_manifest_fingerprint"
    ):
        _check("split_manifest_fingerprint", split_manifest_path)

    # Audit P0-9: the file hashes above can all agree while the shards still
    # belong to a superseded generation of the split bundle -- re-preparing
    # rewrites the bundle and mints a new bundle_id, and only comparing that id
    # catches shards packed from the previous one.
    recorded_bundle = str(manifest.get("source_bundle_id") or "")
    if recorded_bundle:
        current_bundle = _source_bundle_id(current_frame_index)
        if current_bundle and current_bundle != recorded_bundle:
            raise RuntimeError(
                f"Packed data is stale: it was packed from split bundle "
                f"{recorded_bundle}, but the index directory now holds bundle "
                f"{current_bundle}. The split was re-prepared without "
                "repacking, so the shards belong to a superseded generation; "
                "re-run 'cls-trainer dataset pack'."
            )


def verify_packed_shards(manifest_path: str | Path) -> None:
    """Verify every shard's bytes against the manifest hash (audit P0-6).

    ``pack_frame_index`` records a ``shard_sha256`` map, but nothing ever read
    it back, so a truncated, half-written or silently corrupted shard reached
    the DataLoader as garbage pixels and surfaced only as unexplained loss.
    Provenance (which sources the shards came from) and integrity (whether the
    shard bytes are still intact) are separate failures, so this is a separate
    check with its own message.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Packed shard manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Packed manifest is unreadable: {manifest_path}: {exc}"
        ) from exc
    shards = list(manifest.get("shards") or ())
    recorded = manifest.get("shard_sha256")
    if shards and not recorded:
        raise RuntimeError(
            f"Packed manifest {manifest_path} records {len(shards)} shard(s) "
            "but no shard_sha256 map; it predates shard integrity checking. "
            "Re-run 'cls-trainer dataset pack'."
        )
    for name in shards:
        shard_path = manifest_path.parent / name
        if not shard_path.is_file():
            raise RuntimeError(
                f"Packed shard is missing: {shard_path}. The manifest lists "
                f"{len(shards)} shard(s); re-run 'cls-trainer dataset pack'."
            )
        expected = str(recorded.get(name) or "")
        if not expected:
            raise RuntimeError(
                f"Packed manifest {manifest_path} has no shard_sha256 entry "
                f"for {name}; its integrity cannot be verified. Re-run "
                "'cls-trainer dataset pack'."
            )
        actual = _file_sha256(shard_path)
        if actual != expected:
            raise RuntimeError(
                f"Packed shard is corrupt: {name} does not match its recorded "
                f"hash (manifest {expected[:16]}..., current {actual[:16]}...). "
                "The shard changed or was truncated after packing; re-run "
                "'cls-trainer dataset pack'."
            )


class PackedUint8Backend:
    """Read fixed-size CHW uint8 frames by compact integer location."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        image_spec: ImageSpec,
        max_open_shards: int = 16,
    ) -> None:
        try:
            import numpy as np
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Packed backend requires NumPy and pyarrow") from exc
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        image_spec.validate()
        if image_spec.channels != 3:
            raise ValueError(
                "Packed uint8 backend requires RGB images with 3 channels, "
                f"got {image_spec.channels}"
            )
        self.index_path = Path(index_path).resolve()
        self.image_spec = image_spec
        self.channels = image_spec.channels
        self.height = image_spec.height
        self.width = image_spec.width
        self.image_bytes = image_spec.channels * image_spec.height * image_spec.width
        self.max_open_shards = int(max_open_shards)
        schema_names = set(pq.read_schema(self.index_path).names)
        required = {"frame_index", "shard_id", "offset", "length"}
        if not required.issubset(schema_names):
            raise RuntimeError(
                "Legacy path-keyed packed index is unsupported; repack the "
                "dataset to create the integer-index format."
            )
        table = pq.read_table(
            self.index_path,
            columns=["frame_index", "shard_id", "offset", "length"],
        )
        frame_indices = (
            table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False)
        )
        expected = np.arange(len(frame_indices), dtype=frame_indices.dtype)
        if not np.array_equal(frame_indices, expected):
            raise ValueError("Packed frame_index must be contiguous and start at zero")
        self.shard_ids = (
            table["shard_id"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int32, copy=False)
        )
        self.offsets = (
            table["offset"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        self.lengths = (
            table["length"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        manifest_path = self.index_path.with_name("packed_manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Packed shard manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shard_paths = tuple(
            (manifest_path.parent / path).resolve() for path in manifest["shards"]
        )
        manifest_shape = (
            int(manifest["channels"]),
            int(manifest["height"]),
            int(manifest["width"]),
        )
        if manifest_shape != image_spec.chw:
            raise ValueError(
                f"Packed shape {manifest_shape} does not match configured "
                f"shape {image_spec.chw}"
            )
        self._memory_maps: OrderedDict[int, Any] = OrderedDict()

    def _close_map(self, memory_map) -> None:
        mmap_handle = getattr(memory_map, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def _map(self, shard_id: int):
        import numpy as np

        if shard_id in self._memory_maps:
            memory_map = self._memory_maps.pop(shard_id)
            self._memory_maps[shard_id] = memory_map
            return memory_map
        if not 0 <= shard_id < len(self.shard_paths):
            raise IndexError(f"Packed shard id is out of range: {shard_id}")
        memory_map = np.memmap(self.shard_paths[shard_id], mode="r", dtype=np.uint8)
        self._memory_maps[shard_id] = memory_map
        while len(self._memory_maps) > self.max_open_shards:
            _, evicted = self._memory_maps.popitem(last=False)
            self._close_map(evicted)
        return memory_map

    def __call__(self, frame_index: int):
        return self.get_many((frame_index,))[0]

    def get_many(self, frame_indices):
        import numpy as np
        import torch

        locations = [int(frame_index) for frame_index in frame_indices]
        output = np.empty(
            (
                len(locations),
                self.channels,
                self.height,
                self.width,
            ),
            dtype=np.uint8,
        )
        by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_index, location in enumerate(locations):
            if not 0 <= location < len(self.offsets):
                raise IndexError(f"Packed frame index is out of range: {location}")
            length = int(self.lengths[location])
            if length != self.image_bytes:
                raise ValueError(
                    f"Packed image has {length} bytes, expected "
                    f"{self.image_bytes}: frame={location}"
                )
            by_shard[int(self.shard_ids[location])].append((output_index, location))

        for shard_id, items in by_shard.items():
            memory_map = self._map(shard_id)
            for output_index, location in items:
                offset = int(self.offsets[location])
                np.copyto(
                    output[output_index].reshape(-1),
                    memory_map[offset : offset + self.image_bytes],
                )
        return torch.from_numpy(output)

    @property
    def open_shard_count(self) -> int:
        return len(self._memory_maps)

    def close(self) -> None:
        for memory_map in self._memory_maps.values():
            self._close_map(memory_map)
        self._memory_maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_memory_maps"] = OrderedDict()
        return state

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


def pack_frame_index(
    frame_index: str | Path,
    output_dir: str | Path,
    *,
    image_spec: ImageSpec,
    images_per_shard: int = 4096,
    audit_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    source_video_index: str | Path | None = None,
    source_bundle_id: str | None = None,
) -> Path:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Packing requires NumPy, Pillow and pyarrow") from exc
    if images_per_shard <= 0:
        raise ValueError("images_per_shard must be positive")
    image_spec.validate()
    if image_spec.channels != 3:
        raise ValueError(
            f"Packing requires RGB images with 3 channels, got {image_spec.channels}"
        )
    parquet_file = pq.ParquetFile(frame_index)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "packed_frames.parquet"
    index_schema = pa.schema(
        [
            pa.field("frame_index", pa.int64()),
            pa.field("shard_id", pa.int32()),
            pa.field("offset", pa.int64()),
            pa.field("length", pa.int64()),
        ]
    )
    index_writer = pq.ParquetWriter(index_path, index_schema, compression="zstd")
    row_buffer = []
    shard_stream = None
    shard_path = None
    shard_names: list[str] = []
    packed_groups: dict[tuple[str, int, str], list[tuple[int, int]]] = {}
    try:
        index = 0
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if index % images_per_shard == 0:
                    if shard_stream is not None:
                        shard_stream.flush()
                        shard_stream.close()
                    shard_id = index // images_per_shard
                    shard_name = f"shard_{shard_id:06d}.bin"
                    shard_names.append(shard_name)
                    shard_path = output_dir / shard_name
                    shard_stream = shard_path.open("wb")
                else:
                    shard_id = index // images_per_shard
                with Image.open(row["path"]) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if tuple(array.shape) != (
                    image_spec.height,
                    image_spec.width,
                    image_spec.channels,
                ):
                    raise ValueError(
                        f"Unexpected image shape {array.shape}: {row['path']}"
                    )
                chw = np.ascontiguousarray(array.transpose(2, 0, 1))
                # The first frame always opens a shard (index == 0 hits the
                # ``if`` branch), so the stream is non-None here.
                assert shard_stream is not None
                offset = shard_stream.tell()
                shard_stream.write(chw.tobytes())
                row_buffer.append(
                    {
                        "frame_index": index,
                        "shard_id": shard_id,
                        "offset": offset,
                        "length": chw.nbytes,
                    }
                )
                if all(key in row for key in ("game", "label", "video_id", "frame_id")):
                    packed_groups.setdefault(
                        (
                            str(row["game"]),
                            int(row["label"]),
                            str(row["video_id"]),
                        ),
                        [],
                    ).append((int(row["frame_id"]), index))
                if len(row_buffer) >= 4096:
                    index_writer.write_table(
                        pa.Table.from_pylist(row_buffer, schema=index_schema)
                    )
                    row_buffer.clear()
                index += 1
    finally:
        if shard_stream is not None:
            shard_stream.flush()
            shard_stream.close()
        if row_buffer:
            index_writer.write_table(
                pa.Table.from_pylist(row_buffer, schema=index_schema)
            )
        index_writer.close()
    # Audit P0-6: bind the shards to the exact sources they were generated
    # from, so a regenerated index or audit with stale shards is refused before
    # a DataLoader is built. ``dataset_fingerprint`` and ``audit_fingerprint``
    # are both derived from the audit file: the dataset's audit is its
    # fingerprint, and the audit file hash pins the exact audit revision.
    manifest = {
        "format_version": 3,
        "channels": image_spec.channels,
        "height": image_spec.height,
        "width": image_spec.width,
        "image_bytes": (image_spec.channels * image_spec.height * image_spec.width),
        "frame_count": index,
        "source_frame_index_sha256": _file_sha256(frame_index),
        "dataset_fingerprint": _file_sha256(audit_path),
        "audit_fingerprint": _file_sha256(audit_path),
        "split_manifest_fingerprint": _file_sha256(split_manifest_path),
        "shard_sha256": {
            name: _file_sha256(output_dir / name) for name in shard_names
        },
        "shards": shard_names,
    }
    (output_dir / "packed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if packed_groups:
        from game_cls.data.video_index import (
            VideoEntry,
            read_video_entries_parquet,
            write_video_entries_parquet,
        )

        # Audit P0-5: carry identity across from the source video index rather
        # than recomputing it. Keyed on (game, label, video_id), which is
        # exactly how packed_groups is keyed, so the join is exact.
        source_identity: dict[tuple[str, int, str], VideoEntry] = {}
        if source_video_index is not None and Path(source_video_index).is_file():
            source_identity = {
                (entry.game, int(entry.label), entry.video_id): entry
                for entry in read_video_entries_parquet(source_video_index)
            }

        entries = []
        for (game, label, video_id), values in sorted(packed_groups.items()):
            ordered = sorted(values)
            frame_ids = np.asarray(
                [frame_id for frame_id, _ in ordered], dtype=np.int32
            )
            locations = np.asarray(
                [location for _, location in ordered], dtype=np.int64
            )
            id_set = set(frame_ids.tolist())
            valid = {
                delta: np.asarray(
                    [
                        position
                        for position, frame_id in enumerate(frame_ids)
                        if int(frame_id) + delta in id_set
                    ],
                    dtype=np.int32,
                )
                for delta in (1, 2, 3)
            }
            inherited = source_identity.get((game, label, video_id))
            entries.append(
                VideoEntry(
                    game=game,
                    label=label,
                    video_id=video_id,
                    frame_ids=frame_ids,
                    valid_start_positions=valid,
                    frame_locations=locations,
                    # Audit P0-5: the canonical uid is content-anchored and is
                    # computed once in build_video_entries() from per-frame
                    # content hashes, which the packed frame index does not
                    # carry. Rebuilding VideoEntry without it silently demoted
                    # source_video_uid to the game::label::video_id fallback,
                    # so the packed backend and the PNG backend disagreed on
                    # identity and every sidecar/mining/subtype join keyed on
                    # it drifted. Inherit it (and the sidecar-joined fields)
                    # from the source video index instead of recomputing.
                    canonical_source_video_uid=(
                        inherited.canonical_source_video_uid if inherited else ""
                    ),
                    negative_subtype=(
                        inherited.negative_subtype if inherited else None
                    ),
                    sample_weight=(
                        inherited.sample_weight if inherited else 1.0
                    ),
                )
            )
        write_video_entries_parquet(
            entries, output_dir / "packed_video_entries.parquet"
        )
    return index_path
