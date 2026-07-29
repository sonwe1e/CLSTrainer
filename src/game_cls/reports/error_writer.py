from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable


ERROR_FIELDS = [
    "game",
    "label",
    "video_id",
    "frame0_id",
    "frame1_id",
    "delta",
    "image0_path",
    "image1_path",
    "logit0",
    "logit1",
    "margin",
    "probability_class1",
    "prediction",
    "error_type",
    "checkpoint_step",
    "threshold_band",
]


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Evaluation reports require pyarrow") from exc
    return pa, pc, pq


def _report_schema():
    pa, _, _ = _arrow()
    string_fields = {
        "game",
        "video_id",
        "image0_path",
        "image1_path",
        "error_type",
        "threshold_band",
    }
    integer_fields = {
        "label",
        "frame0_id",
        "frame1_id",
        "delta",
        "prediction",
        "checkpoint_step",
    }
    return pa.schema(
        [
            pa.field(
                field,
                pa.string()
                if field in string_fields
                else pa.int64()
                if field in integer_fields
                else pa.float64(),
            )
            for field in ERROR_FIELDS
        ]
    )


def _write_parquet(rows: list[dict], path: Path) -> None:
    pa, _, pq = _arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{field: row.get(field) for field in ERROR_FIELDS} for row in rows]
    pq.write_table(
        pa.Table.from_pylist(normalized, schema=_report_schema()),
        path,
        compression="zstd",
    )


class EvaluationShardWriter:
    """Incrementally writes evaluation rows without retaining the full test set."""

    def __init__(
        self,
        output_dir: str | Path,
        rank: int,
        *,
        row_group_size: int = 4096,
    ) -> None:
        _, _, pq = _arrow()
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        shard_dir = Path(output_dir) / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        schema = _report_schema()
        self._error_writer = pq.ParquetWriter(
            shard_dir / f"errors_rank_{rank:04d}.parquet",
            schema,
            compression="zstd",
        )
        self._near_writer = pq.ParquetWriter(
            shard_dir / f"near_threshold_rank_{rank:04d}.parquet",
            schema,
            compression="zstd",
        )
        self._row_group_size = int(row_group_size)
        self._error_buffer: list[dict] = []
        self._near_buffer: list[dict] = []
        self._closed = False

    @staticmethod
    def _table(rows: list[dict]):
        pa, _, _ = _arrow()
        normalized = [
            {field: row.get(field) for field in ERROR_FIELDS} for row in rows
        ]
        return pa.Table.from_pylist(normalized, schema=_report_schema())

    def write(self, errors: list[dict], near_threshold: list[dict]) -> None:
        self._error_buffer.extend(errors)
        self._near_buffer.extend(near_threshold)
        self._flush_full_groups(
            self._error_buffer, self._error_writer
        )
        self._flush_full_groups(self._near_buffer, self._near_writer)

    def _flush_full_groups(self, buffer: list[dict], writer) -> None:
        while len(buffer) >= self._row_group_size:
            rows = buffer[: self._row_group_size]
            del buffer[: self._row_group_size]
            writer.write_table(self._table(rows))

    def _flush_remaining(self, buffer: list[dict], writer) -> None:
        if buffer:
            writer.write_table(self._table(buffer))
            buffer.clear()

    def close(self) -> None:
        if not self._closed:
            self._flush_remaining(
                self._error_buffer, self._error_writer
            )
            self._flush_remaining(self._near_buffer, self._near_writer)
            self._error_writer.close()
            self._near_writer.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def prepare_evaluation_directory(output_dir: str | Path, rank: int) -> None:
    """Remove only stale generated shard files before a repeated evaluation."""
    output_dir = Path(output_dir)
    if rank == 0:
        shard_dir = output_dir / "shards"
        if shard_dir.is_dir():
            for pattern in ("errors_rank_*.parquet", "near_threshold_rank_*.parquet"):
                for path in shard_dir.glob(pattern):
                    path.unlink()


def write_evaluation_shard(
    output_dir: str | Path,
    rank: int,
    errors: list[dict],
    near_threshold: list[dict],
) -> None:
    shard_dir = Path(output_dir) / "shards"
    _write_parquet(errors, shard_dir / f"errors_rank_{rank:04d}.parquet")
    _write_parquet(
        near_threshold, shard_dir / f"near_threshold_rank_{rank:04d}.parquet"
    )


def _write_group_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _image_cell(path: str) -> str:
    if not path:
        return "<span>no image path</span>"
    if path.startswith("packed://"):
        return "<span>packed preview unavailable</span>"
    escaped = html.escape(path, quote=True)
    return (
        f"<a href='{escaped}'><img src='{escaped}' loading='lazy' "
        "style='max-width:100%;max-height:320px'></a>"
    )


def _write_html(path: Path, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<article><div class='images'>"
            + _image_cell(str(row.get("image0_path", "")))
            + _image_cell(str(row.get("image1_path", "")))
            + "</div><pre>"
            + html.escape(json.dumps(row, ensure_ascii=False, indent=2))
            + "</pre></article>"
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Evaluation errors</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#f5f6f8}"
        "article{background:white;padding:16px;margin:16px 0;border-radius:10px}"
        ".images{display:flex;gap:12px;align-items:flex-start}pre{white-space:pre-wrap}"
        "</style><h1>FP / FN 双帧报告</h1>"
        + "".join(cards),
        encoding="utf-8",
    )


def _materialize_packed_previews(
    output_dir: Path,
    rows: list[dict],
    decoder,
) -> list[dict]:
    if decoder is None:
        return rows
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Packed HTML previews require NumPy and Pillow"
        ) from exc
    preview_dir = output_dir / "previews"
    rewritten = []
    exported: dict[int, str] = {}
    for original in rows:
        row = dict(original)
        for field in ("image0_path", "image1_path"):
            reference = str(row.get(field, ""))
            prefix = "packed://frame/"
            if not reference.startswith(prefix):
                continue
            frame_index = int(reference.removeprefix(prefix))
            relative = exported.get(frame_index)
            if relative is None:
                preview_dir.mkdir(parents=True, exist_ok=True)
                relative = f"previews/frame_{frame_index:09d}.png"
                tensor = decoder(frame_index)
                array = tensor.detach().cpu().permute(1, 2, 0).numpy()
                Image.fromarray(np.asarray(array, dtype=np.uint8)).save(
                    output_dir / relative
                )
                exported[frame_index] = relative
            row[field] = relative
        rewritten.append(row)
    return rewritten


def _merge_error_shards(
    paths: Iterable[Path],
    false_positive_path: Path,
    false_negative_path: Path,
    *,
    html_max_errors: int,
) -> list[dict]:
    pa, pc, pq = _arrow()
    schema = _report_schema()
    fp_writer = pq.ParquetWriter(false_positive_path, schema, compression="zstd")
    fn_writer = pq.ParquetWriter(false_negative_path, schema, compression="zstd")
    preview: list[dict] = []
    try:
        for path in sorted(paths):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch], schema=schema)
                for error_type, writer in (("FP", fp_writer), ("FN", fn_writer)):
                    filtered = table.filter(
                        pc.equal(table["error_type"], pa.scalar(error_type))
                    )
                    if len(filtered):
                        writer.write_table(filtered)
                        remaining = html_max_errors - len(preview)
                        if remaining > 0:
                            preview.extend(
                                filtered.slice(0, remaining).to_pylist()
                            )
    finally:
        fp_writer.close()
        fn_writer.close()
    return preview


def _merge_plain_shards(paths: Iterable[Path], output_path: Path) -> None:
    pa, _, pq = _arrow()
    schema = _report_schema()
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    try:
        for path in sorted(paths):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch], schema=schema)
                if len(table):
                    writer.write_table(table)
    finally:
        writer.close()


def write_evaluation_report(
    output_dir: str | Path,
    metrics: dict,
    grouped_metrics: dict | None = None,
    *,
    errors: list[dict] | None = None,
    near_threshold: list[dict] | None = None,
    merge_shards: bool = False,
    html_max_errors: int = 200,
    lightweight: bool = False,
    preview_decoder=None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_metrics = grouped_metrics or {}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not lightweight:
        _write_group_csv(
            output_dir / "metrics_by_game.csv",
            grouped_metrics.get("by_game", []),
        )
        _write_group_csv(
            output_dir / "metrics_by_video.csv",
            grouped_metrics.get("by_video", []),
        )
        _write_group_csv(
            output_dir / "metrics_by_game_label.csv",
            grouped_metrics.get("by_game_label", []),
        )
    if merge_shards:
        preview = _merge_error_shards(
            (output_dir / "shards").glob("errors_rank_*.parquet"),
            output_dir / "false_positive.parquet",
            output_dir / "false_negative.parquet",
            html_max_errors=html_max_errors,
        )
        _merge_plain_shards(
            (output_dir / "shards").glob("near_threshold_rank_*.parquet"),
            output_dir / "near_threshold.parquet",
        )
    else:
        errors = errors or []
        preview = errors[:html_max_errors]
        _write_parquet(
            [row for row in errors if row.get("error_type") == "FP"],
            output_dir / "false_positive.parquet",
        )
        _write_parquet(
            [row for row in errors if row.get("error_type") == "FN"],
            output_dir / "false_negative.parquet",
        )
        _write_parquet(
            near_threshold or [], output_dir / "near_threshold.parquet"
        )
    if not lightweight:
        preview = _materialize_packed_previews(
            output_dir, preview, preview_decoder
        )
        _write_html(output_dir / "errors.html", preview)
