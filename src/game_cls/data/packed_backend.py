from __future__ import annotations

import contextlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from .image_spec import ImageSpec


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
    manifest = {
        "format_version": 2,
        "channels": image_spec.channels,
        "height": image_spec.height,
        "width": image_spec.width,
        "image_bytes": (image_spec.channels * image_spec.height * image_spec.width),
        "frame_count": index,
        "shards": shard_names,
    }
    (output_dir / "packed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if packed_groups:
        from game_cls.data.video_index import (
            VideoEntry,
            write_video_entries_parquet,
        )

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
            entries.append(
                VideoEntry(
                    game=game,
                    label=label,
                    video_id=video_id,
                    frame_ids=frame_ids,
                    valid_start_positions=valid,
                    frame_locations=locations,
                )
            )
        write_video_entries_parquet(
            entries, output_dir / "packed_video_entries.parquet"
        )
    return index_path
