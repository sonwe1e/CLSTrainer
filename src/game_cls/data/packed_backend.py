from __future__ import annotations

from pathlib import Path
from typing import Any


class PackedUint8Backend:
    """Reads fixed-size CHW uint8 images from memory-mapped shard files."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        channels: int = 3,
        height: int = 448,
        width: int = 208,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Packed backend requires pyarrow") from exc
        self.channels = channels
        self.height = height
        self.width = width
        self.image_bytes = channels * height * width
        rows = pq.read_table(
            index_path,
            columns=["path", "shard_path", "offset", "length"],
        ).to_pylist()
        self.locations = {
            row["path"]: (
                row["shard_path"],
                int(row["offset"]),
                int(row["length"]),
            )
            for row in rows
        }
        self._memory_maps: dict[str, Any] = {}

    def _map(self, path: str):
        import numpy as np

        if path not in self._memory_maps:
            self._memory_maps[path] = np.memmap(path, mode="r", dtype=np.uint8)
        return self._memory_maps[path]

    def __call__(self, path: str):
        import numpy as np
        import torch

        if path not in self.locations:
            raise KeyError(f"Image path is absent from packed index: {path}")
        shard_path, offset, length = self.locations[path]
        if length != self.image_bytes:
            raise ValueError(
                f"Packed image has {length} bytes, expected {self.image_bytes}: {path}"
            )
        array = np.asarray(
            self._map(shard_path)[offset : offset + length]
        ).reshape(self.channels, self.height, self.width)
        return torch.from_numpy(array.copy())

    def close(self) -> None:
        for memory_map in self._memory_maps.values():
            mmap_handle = getattr(memory_map, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        self._memory_maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_memory_maps"] = {}
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def pack_frame_index(
    frame_index: str | Path,
    output_dir: str | Path,
    *,
    images_per_shard: int = 4096,
    expected_width: int = 208,
    expected_height: int = 448,
) -> Path:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Packing requires NumPy, Pillow and pyarrow"
        ) from exc
    if images_per_shard <= 0:
        raise ValueError("images_per_shard must be positive")
    parquet_file = pq.ParquetFile(frame_index)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "packed_frames.parquet"
    index_schema = pa.schema(
        [
            pa.field("path", pa.string()),
            pa.field("shard_path", pa.string()),
            pa.field("offset", pa.int64()),
            pa.field("length", pa.int64()),
        ]
    )
    index_writer = pq.ParquetWriter(
        index_path, index_schema, compression="zstd"
    )
    row_buffer = []
    shard_stream = None
    shard_path = None
    try:
        index = 0
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if index % images_per_shard == 0:
                    if shard_stream is not None:
                        shard_stream.flush()
                        shard_stream.close()
                    shard_path = (
                        output_dir
                        / f"shard_{index // images_per_shard:06d}.bin"
                    )
                    shard_stream = shard_path.open("wb")
                with Image.open(row["path"]) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if tuple(array.shape) != (expected_height, expected_width, 3):
                    raise ValueError(
                        f"Unexpected image shape {array.shape}: {row['path']}"
                    )
                chw = np.ascontiguousarray(array.transpose(2, 0, 1))
                offset = shard_stream.tell()
                shard_stream.write(chw.tobytes())
                row_buffer.append(
                    {
                        "path": row["path"],
                        "shard_path": str(shard_path),
                        "offset": offset,
                        "length": chw.nbytes,
                    }
                )
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
    return index_path
