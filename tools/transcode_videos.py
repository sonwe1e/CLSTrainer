#!/usr/bin/env python3
"""Incrementally build training-friendly 448x208 videos from source videos.

Resize pipeline (exactly as requested):

    source frame
      -> torch bilinear resize to 832x1792   (4x target H/W)
      -> torch bilinear resize to 416x896    (2x target H/W)
      -> DPID 2x downsample, lambda=3
      -> 208x448 RGB uint8

The source directory tree is mirrored under the output tree.  A manifest makes
reruns incremental: already completed videos are skipped when the source
identity, output file and processing/encoding signature still match.  New or
changed source videos are processed automatically.

The tool needs PyTorch plus ffmpeg/ffprobe executables; no PyAV/OpenCV package
is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch
import torch.nn.functional as F


PIPELINE_VERSION = 1
TARGET_H = 208
TARGET_W = 448
STAGE1_H = TARGET_H * 4
STAGE1_W = TARGET_W * 4
STAGE2_H = TARGET_H * 2
STAGE2_W = TARGET_W * 2
DEFAULT_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts")


@dataclass(frozen=True)
class Task:
    split: str
    source_root: str
    output_root: str
    source_path: str
    source_rel: str
    output_rel: str


@dataclass(frozen=True)
class WorkerOptions:
    ffmpeg: str
    ffprobe: str
    device: str
    batch_frames: int
    torch_threads: int
    decoder_threads: int
    encoder_threads: int
    codec: str
    preset: str
    crf: int
    pix_fmt: str
    gop: int
    dpid_lambda: float
    bilinear_antialias: bool


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pipeline_payload(args: argparse.Namespace) -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "resize": {
            "stage1": [STAGE1_H, STAGE1_W],
            "stage1_mode": "torch_bilinear",
            "stage2": [STAGE2_H, STAGE2_W],
            "stage2_mode": "torch_bilinear",
            "bilinear_align_corners": False,
            "bilinear_antialias": bool(args.bilinear_antialias),
            "stage3": [TARGET_H, TARGET_W],
            "stage3_mode": "dpid_2x",
            "dpid_lambda": float(args.dpid_lambda),
        },
        "encoding": {
            "container": "mp4",
            "codec": args.codec,
            "preset": args.preset,
            "crf": int(args.crf),
            "pix_fmt": args.pix_fmt,
            "gop": int(args.gop),
            "audio": False,
        },
    }


def _sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path, mode: str) -> dict:
    stat = path.stat()
    identity = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if mode == "sha256":
        identity["sha256"] = _sha256(path)
    return identity


def output_identity(path: Path) -> dict:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_manifest(output_root: Path) -> dict:
    path = output_root / ".transcode_manifest.json"
    if not path.is_file():
        return {"version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries", {}), dict):
        raise RuntimeError(f"Invalid manifest format: {path}")
    payload.setdefault("version", 1)
    payload.setdefault("entries", {})
    return payload


def _write_manifest(output_root: Path, manifest: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ".transcode_manifest.json"
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=output_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _extensions(values: Iterable[str]) -> set[str]:
    result = set()
    for value in values:
        value = value.strip().lower()
        if not value:
            continue
        result.add(value if value.startswith(".") else "." + value)
    return result


def discover_tasks(
    *,
    split: str,
    source_root: Path,
    output_root: Path,
    extensions: set[str],
) -> list[Task]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"{split} source root does not exist: {source_root}")
    tasks: list[Task] = []
    output_seen: dict[str, str] = {}
    for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(source_root)
        output_rel = rel.with_suffix(".mp4").as_posix()
        previous = output_seen.get(output_rel)
        if previous is not None:
            raise ValueError(
                f"Output collision: both {previous} and {rel.as_posix()} map to {output_rel}"
            )
        output_seen[output_rel] = rel.as_posix()
        tasks.append(
            Task(
                split=split,
                source_root=str(source_root),
                output_root=str(output_root),
                source_path=str(path),
                source_rel=rel.as_posix(),
                output_rel=output_rel,
            )
        )
    return tasks


def should_skip(
    task: Task,
    *,
    manifest: dict,
    pipeline_hash: str,
    fingerprint_mode: str,
    force: bool,
) -> tuple[bool, dict]:
    source = Path(task.source_path)
    current_source = source_identity(source, fingerprint_mode)
    if force:
        return False, current_source
    entry = manifest.get("entries", {}).get(task.output_rel)
    if not isinstance(entry, dict):
        return False, current_source
    if entry.get("pipeline_hash") != pipeline_hash:
        return False, current_source
    if entry.get("source", {}).get("relative_path") != task.source_rel:
        return False, current_source
    recorded_source = entry.get("source", {}).get("identity")
    if recorded_source != current_source:
        return False, current_source

    output_path = Path(task.output_root) / task.output_rel
    if not output_path.is_file():
        return False, current_source
    recorded_output = entry.get("output", {}).get("identity")
    if recorded_output != output_identity(output_path):
        return False, current_source
    return True, current_source


def _probe(path: Path, ffprobe: str) -> dict:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,pix_fmt,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream in {path}")
    stream = streams[0]
    fps = Fraction(str(stream.get("avg_frame_rate", "0/0")))
    if fps <= 0:
        raise ValueError(f"Invalid avg_frame_rate for {path}: {stream.get('avg_frame_rate')}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "r_frame_rate": str(stream.get("r_frame_rate", "")),
        "pix_fmt": str(stream.get("pix_fmt", "")),
        "codec_name": str(stream.get("codec_name", "")),
        "duration": float(payload.get("format", {}).get("duration", 0.0) or 0.0),
    }


def _read_up_to(handle: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        block = handle.read(remaining)
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def dpid_downsample_2x(images: torch.Tensor, *, lambda_: float = 3.0) -> torch.Tensor:
    """Vectorized exact-2x form of the DPID reference algorithm.

    ``images`` is ``[N,3,2H,2W]`` float in the 0..255 range.  The implementation
    follows the reference algorithm's area-average image, 3x3 [1 2 1; 2 4 2;
    1 2 1] local reference, and normalized RGB-distance^lambda detail weights.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [N,3,H,W], got {tuple(images.shape)}")
    if images.shape[-2] % 2 or images.shape[-1] % 2:
        raise ValueError("DPID 2x requires even input height and width")
    if lambda_ < 0:
        raise ValueError("DPID lambda must be >= 0")

    avg_image = F.avg_pool2d(images, kernel_size=2, stride=2)
    kernel = images.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    color_kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    numerator = F.conv2d(avg_image, color_kernel, padding=1, groups=3)
    denominator = F.conv2d(
        torch.ones_like(avg_image[:, :1]), kernel.view(1, 1, 3, 3), padding=1
    )
    reference = numerator / denominator

    n, c, out_h, out_w = avg_image.shape
    pixels = (
        images.reshape(n, c, out_h, 2, out_w, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )  # [N,3,H,W,2,2]
    if lambda_ == 0:
        weights = torch.ones(
            (n, out_h, out_w, 2, 2), dtype=images.dtype, device=images.device
        )
    else:
        distance = torch.linalg.vector_norm(
            pixels - reference[..., None, None], ord=2, dim=1
        ) / (255.0 * math.sqrt(3.0))
        weights = distance.pow(float(lambda_))

    weight_sum = weights.sum(dim=(-1, -2))  # [N,H,W]
    weighted = (pixels * weights[:, None]).sum(dim=(-1, -2))
    safe = weighted / weight_sum.clamp_min(torch.finfo(images.dtype).eps)[:, None]
    return torch.where((weight_sum > 0)[:, None], safe, reference)


def resize_pipeline(
    frames: torch.Tensor,
    *,
    dpid_lambda: float,
    bilinear_antialias: bool,
) -> torch.Tensor:
    """Apply the requested three-stage resize and return uint8 [N,3,208,448]."""

    x = frames.float()
    x = F.interpolate(
        x,
        size=(STAGE1_H, STAGE1_W),
        mode="bilinear",
        align_corners=False,
        antialias=bilinear_antialias,
    )
    x = F.interpolate(
        x,
        size=(STAGE2_H, STAGE2_W),
        mode="bilinear",
        align_corners=False,
        antialias=bilinear_antialias,
    )
    x = dpid_downsample_2x(x, lambda_=dpid_lambda)
    return x.round().clamp_(0, 255).to(torch.uint8)


def _device(name: str) -> torch.device:
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("--device npu requires torch_npu") from exc
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "npu" and not torch.npu.is_available():
        raise RuntimeError("NPU requested but unavailable")
    return device


def transcode_one(task: Task, options: WorkerOptions, source_id: dict, pipeline_hash: str) -> dict:
    torch.set_num_threads(max(1, int(options.torch_threads)))
    source = Path(task.source_path)
    output_root = Path(task.output_root)
    output = output_root / task.output_rel
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _probe(source, options.ffprobe)
    fps: Fraction = metadata["fps"]
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_bytes = width * height * 3
    if frame_bytes <= 0:
        raise ValueError(f"Invalid source size for {source}: {height}x{width}")

    device = _device(options.device)
    temp = output.with_name(f".{output.stem}.partial.{os.getpid()}.{time.time_ns()}.mp4")
    decoder_cmd = [
        options.ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-threads",
        str(options.decoder_threads),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    encoder_cmd = [
        options.ffmpeg,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{TARGET_W}x{TARGET_H}",
        "-r",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        options.codec,
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-pix_fmt",
        options.pix_fmt,
        "-g",
        str(options.gop),
        "-keyint_min",
        str(options.gop),
        "-sc_threshold",
        "0",
        "-threads",
        str(options.encoder_threads),
        "-movflags",
        "+faststart",
        str(temp),
    ]

    decoder = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_written = 0
    started = time.time()
    try:
        assert decoder.stdout is not None
        assert encoder.stdin is not None
        batch_bytes = frame_bytes * int(options.batch_frames)
        while True:
            raw = _read_up_to(decoder.stdout, batch_bytes)
            if not raw:
                break
            if len(raw) % frame_bytes:
                raise RuntimeError(
                    f"Decoder returned a partial RGB frame for {source}: {len(raw) % frame_bytes} bytes"
                )
            count = len(raw) // frame_bytes
            array = np.frombuffer(raw, dtype=np.uint8).copy().reshape(count, height, width, 3)
            batch = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().to(device)
            with torch.no_grad():
                resized = resize_pipeline(
                    batch,
                    dpid_lambda=options.dpid_lambda,
                    bilinear_antialias=options.bilinear_antialias,
                )
            rgb = resized.permute(0, 2, 3, 1).contiguous().cpu().numpy().tobytes()
            encoder.stdin.write(rgb)
            frames_written += count

        decoder.stdout.close()
        decoder_stderr = decoder.stderr.read() if decoder.stderr is not None else b""
        decoder_code = decoder.wait()
        if decoder_code != 0:
            raise RuntimeError(
                f"Decoder failed ({decoder_code}) for {source}: "
                + decoder_stderr.decode("utf-8", errors="replace")[-4000:]
            )

        encoder.stdin.close()
        encoder_stderr = encoder.stderr.read() if encoder.stderr is not None else b""
        encoder_code = encoder.wait()
        if encoder_code != 0:
            raise RuntimeError(
                f"Encoder failed ({encoder_code}) for {source}: "
                + encoder_stderr.decode("utf-8", errors="replace")[-4000:]
            )
        if frames_written <= 0 or not temp.is_file() or temp.stat().st_size <= 0:
            raise RuntimeError(f"No frames were written for {source}")

        verified = _probe(temp, options.ffprobe)
        if (int(verified["height"]), int(verified["width"])) != (TARGET_H, TARGET_W):
            raise RuntimeError(f"Output verification failed for {temp}: wrong resolution")
        os.replace(temp, output)
    except BaseException:
        for process in (decoder, encoder):
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=2)
            except Exception:
                pass
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise

    elapsed = time.time() - started
    stat = output.stat()
    return {
        "pipeline_hash": pipeline_hash,
        "source": {
            "relative_path": task.source_rel,
            "identity": source_id,
            "width": width,
            "height": height,
            "fps_num": fps.numerator,
            "fps_den": fps.denominator,
            "r_frame_rate": metadata["r_frame_rate"],
            "codec": metadata["codec_name"],
            "pix_fmt": metadata["pix_fmt"],
        },
        "output": {
            "relative_path": task.output_rel,
            "identity": {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)},
            "frames": int(frames_written),
            "width": TARGET_W,
            "height": TARGET_H,
            "fps_num": fps.numerator,
            "fps_den": fps.denominator,
            "codec": options.codec,
            "pix_fmt": options.pix_fmt,
            "gop": int(options.gop),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
    }


def _resolve_output_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train_output_root or args.test_output_root:
        if not args.train_output_root or not args.test_output_root:
            raise ValueError("Specify both --train-output-root and --test-output-root")
        return Path(args.train_output_root).resolve(), Path(args.test_output_root).resolve()
    if not args.output_root:
        raise ValueError("Use --output-root or both split-specific output roots")
    root = Path(args.output_root).resolve()
    return root / "train", root / "test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument(
        "--output-root",
        help="Creates <output-root>/train and <output-root>/test while preserving relative paths.",
    )
    parser.add_argument("--train-output-root")
    parser.add_argument("--test-output-root")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS))

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-frames", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--decoder-threads", type=int, default=2)
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, npu:0 ...")

    parser.add_argument("--dpid-lambda", type=float, default=3.0)
    parser.add_argument("--bilinear-antialias", action="store_true")
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--pix-fmt", default="yuv444p")
    parser.add_argument("--gop", type=int, default=12)

    parser.add_argument(
        "--fingerprint",
        choices=("size_mtime", "sha256"),
        default="size_mtime",
        help="sha256 is stronger but rereads every source video on every scan.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even matching manifest entries.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("workers", "batch_frames", "torch_threads", "decoder_threads", "encoder_threads", "gop"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.dpid_lambda < 0:
        raise ValueError("--dpid-lambda must be >= 0")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be in [0,51]")
    for binary in (args.ffmpeg if hasattr(args, "ffmpeg") else "ffmpeg", args.ffprobe if hasattr(args, "ffprobe") else "ffprobe"):
        if shutil.which(binary) is None:
            raise FileNotFoundError(f"Required executable not found on PATH: {binary}")


def main() -> int:
    parser = build_parser()
    # Keep executable names configurable without cluttering the normal help.
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    _validate_args(args)

    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    train_output, test_output = _resolve_output_roots(args)
    extensions = _extensions(args.extensions)

    pipeline = pipeline_payload(args)
    pipeline_hash = _json_hash(pipeline)
    print("pipeline_hash:", pipeline_hash)
    print("resize: source -> 832x1792 bilinear -> 416x896 bilinear -> 208x448 DPID lambda=", args.dpid_lambda)
    print("train_output:", train_output)
    print("test_output :", test_output)

    tasks = discover_tasks(
        split="train", source_root=train_root, output_root=train_output, extensions=extensions
    ) + discover_tasks(
        split="test", source_root=test_root, output_root=test_output, extensions=extensions
    )
    if not tasks:
        print("No source videos found.", file=sys.stderr)
        return 2

    manifests = {
        str(train_output): _load_manifest(train_output),
        str(test_output): _load_manifest(test_output),
    }
    for output_root, manifest in manifests.items():
        manifest["version"] = 1
        manifest["pipeline"] = pipeline
        manifest["pipeline_hash"] = pipeline_hash
        manifest["source_fingerprint_mode"] = args.fingerprint
        manifest.setdefault("entries", {})

    pending: list[tuple[Task, dict]] = []
    skipped = 0
    for task in tasks:
        manifest = manifests[str(Path(task.output_root))]
        skip, source_id = should_skip(
            task,
            manifest=manifest,
            pipeline_hash=pipeline_hash,
            fingerprint_mode=args.fingerprint,
            force=bool(args.force),
        )
        if skip:
            skipped += 1
        else:
            pending.append((task, source_id))

    print(f"discovered={len(tasks)} skipped={skipped} pending={len(pending)}")
    if args.dry_run:
        for task, _ in pending[:100]:
            print(f"PLAN {task.split:5s} {task.source_rel} -> {task.output_rel}")
        if len(pending) > 100:
            print(f"... {len(pending) - 100} more")
        return 0
    if not pending:
        print("Everything is up to date.")
        return 0

    options = WorkerOptions(
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        device=args.device,
        batch_frames=args.batch_frames,
        torch_threads=args.torch_threads,
        decoder_threads=args.decoder_threads,
        encoder_threads=args.encoder_threads,
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
        pix_fmt=args.pix_fmt,
        gop=args.gop,
        dpid_lambda=args.dpid_lambda,
        bilinear_antialias=bool(args.bilinear_antialias),
    )

    completed = 0
    failures: list[tuple[Task, str]] = []

    def record_success(task: Task, entry: dict) -> None:
        nonlocal completed
        output_root = Path(task.output_root)
        manifest = manifests[str(output_root)]
        manifest["entries"][task.output_rel] = entry
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(output_root, manifest)
        completed += 1
        fps = entry["output"]["fps_num"] / entry["output"]["fps_den"]
        print(
            f"OK   [{completed}/{len(pending)}] {task.split:5s} {task.source_rel} "
            f"frames={entry['output']['frames']} fps={fps:.3f} "
            f"time={entry['elapsed_seconds']:.1f}s"
        )

    if args.workers == 1:
        for task, source_id in pending:
            try:
                record_success(task, transcode_one(task, options, source_id, pipeline_hash))
            except Exception as exc:
                failures.append((task, repr(exc)))
                print(f"FAIL {task.split:5s} {task.source_rel}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    break
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(transcode_one, task, options, source_id, pipeline_hash): task
                for task, source_id in pending
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    record_success(task, future.result())
                except Exception as exc:
                    failures.append((task, repr(exc)))
                    print(f"FAIL {task.split:5s} {task.source_rel}: {exc}", file=sys.stderr)
                    if args.fail_fast:
                        for other in future_map:
                            other.cancel()
                        break

    print(
        f"done: discovered={len(tasks)} skipped={skipped} completed={completed} failed={len(failures)}"
    )
    if failures:
        for task, error in failures[:20]:
            print(f"  {task.split}/{task.source_rel}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
