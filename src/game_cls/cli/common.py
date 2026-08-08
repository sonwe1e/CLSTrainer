"""CLSTrainer command implementation for contract 5."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

DEFAULT_RUNS_ROOT = "runs"


_MAX_TEE_BUFFER_BYTES = 8 * 1024 * 1024


class _StreamTee:
    """Mirror a stream while capturing it for ``console.log``.

    The run directory only becomes known after allocation, so output is
    buffered until ``attach`` is called, then streamed to the file.
    """

    def __init__(self, stream: TextIO):
        self._stream = stream
        self._buffer: list[str] = []
        self._buffered = 0
        self._file: TextIO | None = None

    def write(self, text: str) -> int:
        self._stream.write(text)
        if self._file is not None:
            self._file.write(text)
            return len(text)
        self._buffer.append(text)
        self._buffered += len(text)
        if self._buffered > _MAX_TEE_BUFFER_BYTES:
            dropped = self._buffer.pop(0)
            self._buffered -= len(dropped)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        if self._file is not None:
            self._file.flush()

    def attach(self, run_dir: Path) -> None:
        if self._file is not None:
            return
        self._file = (Path(run_dir) / "console.log").open("a", encoding="utf-8")
        self._file.write("".join(self._buffer))
        self._buffer.clear()
        self._buffered = 0
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def isatty(self) -> bool:
        return False


class _TeeContext:
    def __init__(self) -> None:
        self.stdout = _StreamTee(sys.stdout)
        self.stderr = _StreamTee(sys.stderr)

    def attach(self, run_dir: Path) -> None:
        self.stdout.attach(run_dir)
        self.stderr.attach(run_dir)

    def __enter__(self) -> _TeeContext:
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.stdout, self.stderr
        return self

    def __exit__(self, *exc_info: Any) -> None:
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
        self.stdout.close()
        self.stderr.close()


# ---------------------------------------------------------------------------
# resume drift check + run resolution helpers
# ---------------------------------------------------------------------------

# Differences that are normal when resuming (never warned about).
RESUME_EXPECTED_DIFFS = {
    "experiment.output_dir",
    "train.resume_path",
    "checkpoint.save_last_every_steps",
}


# Changing max_steps/stop_after does not corrupt the restored optimizer/
# sampler state; it only re-plans the scheduler, which is why it is
# classified as ``resume-extend`` rather than exact.
RESUME_EXTEND_KEYS = {"train.max_steps", "train.stop_after_steps"}


# Facts that must never change across a resume; they would silently
# invalidate the restored optimizer/sampler/model state or change what the
# run measures. Changing any of these is ``fork`` territory.
RESUME_CRITICAL_DIFFS = {
    "decision.threshold",
    "data.width",
    "data.height",
    "data.channels",
    "pair.test_delta",
    "data.backend",
    "data.train_index",
    "data.val_index",
    "data.test_index",
    "data.train_video_index",
    "data.val_video_index",
    "data.test_video_index",
    "data.train_packed_index",
    "data.val_packed_index",
    "data.test_packed_index",
    "data.train_packed_video_index",
    "data.val_packed_video_index",
    "data.test_packed_video_index",
    "sampler.class_probability",
    "pair.train_delta_probability",
    "sampler.game_alpha",
    "model.factory",
    "model.trainable_rules",
    "model.num_classes",
    "model.checkpoint_path",
    "model.require_pretrained_backbone",
    "experiment.seed",
    "train.local_batch_size",
    "optimizer.learning_rate",
    "optimizer.weight_decay",
    "distributed.enabled",
    "distributed.backend",
}


def _flatten_dict(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


def check_resume_drift(
    baseline: dict[str, Any], config: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Compare a resume config against the run's resolved config.

    Returns (critical, warnings) lists of human readable problems. The
    comparison is symmetric: added and removed keys are detected, not just
    changed values, so newly introduced critical fields cannot silently
    slip past.
    """
    base_flat = _flatten_dict(baseline)
    new_flat = _flatten_dict(config)
    critical: list[str] = []
    warnings: list[str] = []
    for key in sorted(set(base_flat) | set(new_flat)):
        in_base = key in base_flat
        in_new = key in new_flat
        if in_base and in_new:
            if base_flat[key] == new_flat[key]:
                continue
            text = f"{key}: {base_flat[key]!r} -> {new_flat[key]!r}"
        elif in_base:
            text = f"{key}: removed ({base_flat[key]!r})"
        else:
            text = f"{key}: added ({new_flat[key]!r})"
        if key in RESUME_EXPECTED_DIFFS and in_base and in_new:
            continue
        if key in RESUME_CRITICAL_DIFFS:
            critical.append(text)
        else:
            warnings.append(text)
    return critical, warnings


def classify_resume(baseline: dict, config: dict, warnings: list[str]) -> str:
    """exact | extend | fork for a resume with non-critical drift."""
    if not warnings:
        return "exact"
    flexible_only = all(
        warning.split(":", 1)[0].strip() in RESUME_EXTEND_KEYS for warning in warnings
    )
    return "extend" if flexible_only else "fork"


def _resolve_run_dir(target: str, root: Path) -> Path:
    if target == "latest":
        records = _find_index_records(root)
        if records:
            return Path(records[-1]["output_dir"])
        candidates = sorted(
            path for path in root.glob("*/*") if (path / "status.json").is_file()
        )
        if candidates:
            return candidates[-1]
        raise SystemExit(f"No runs found under {root.resolve()}.")
    path = Path(target)
    if path.is_dir():
        return path.resolve()
    for record in _find_index_records(root):
        if record.get("run_id") == target:
            return Path(record["output_dir"])
    raise SystemExit(
        f"Run not found: {target!r} (not a directory and not in the index "
        f"under {root.resolve()})."
    )


def _read_run_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _resolve_resume_checkpoint(resume_dir: Path, config: dict) -> str:
    explicit = config["train"].get("resume_path")
    if explicit:
        return str(explicit)
    candidate = resume_dir / "checkpoints" / "checkpoint_last.pth"
    if not candidate.is_file():
        raise SystemExit(
            f"--resume {resume_dir} has no checkpoints/checkpoint_last.pth; "
            "pass train.resume_path explicitly."
        )
    return str(candidate)


# ---------------------------------------------------------------------------
# run list / show
# ---------------------------------------------------------------------------


def _find_index_records(root: Path) -> list[dict[str, Any]]:
    """All index records under ``root``, aggregated by run identity.

    The index is append-only (every start/resume/finish appends a line);
    readers must collapse per-``run_id`` to the latest record so a resumed
    run shows its current state, not the pre-resume one.
    """
    from game_cls.runs import read_run_index

    records: list[dict[str, Any]] = []
    records.extend(read_run_index(root))
    for nested in sorted(root.glob("*/index.jsonl")):
        records.extend(read_run_index(nested.parent))
    latest_index: dict[str, int] = {}
    for index, record in enumerate(records):
        key = record.get("run_id") or record.get("output_dir")
        if key:
            latest_index[key] = index
    aggregated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        key = record.get("run_id") or record.get("output_dir")
        if key and latest_index.get(key) != index:
            continue  # superseded by a later record for the same run
        aggregated.append(record)
    return aggregated


# ---------------------------------------------------------------------------
# evaluate (standalone final test / validation evaluation)
# ---------------------------------------------------------------------------


_CHECKPOINT_ALIASES = (
    "last",
    "best_selection",
    "best_val_loss",
    "best_worst_game",
)
