"""Run management: immutable run directories, manifests and status.

Every training start is modeled as an independent, non-overwritable Run:

* ``allocate_run_dir`` creates a dated, timestamped directory. The
  ``mkdir(exist_ok=False)`` guarantees two runs can never land in the same
  directory.
* ``manifest.json`` records everything needed to explain a run later:
  command, git fingerprint, environment, seed, world size, resume parent.
* ``status.json`` is atomically updated during training so an aborted run
  still tells you whether it is RUNNING, SUCCEEDED or FAILED.
* ``index.jsonl`` at the runs root allows cheap listing/comparison.
* ``summary.md`` is the one-page human readable result.
"""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_RUNNING = "RUNNING"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"


def _now() -> datetime:
    return datetime.now().astimezone()


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name).strip()).strip("-")
    return slug.lower() or "run"


def allocate_run_dir(
    runs_root: str | Path, name: str, now: datetime | None = None
) -> tuple[Path, str]:
    """Create a brand-new run directory; never reuse or overwrite one.

    Layout: ``<runs_root>/<YYYYmmdd>/<HHMMSS>_<slug>_<rand>/``.
    """
    moment = now or _now()
    runs_root = Path(runs_root)
    date_dir = runs_root / moment.strftime("%Y%m%d")
    slug = slugify(name)
    for _ in range(64):
        run_id = f"{moment.strftime('%H%M%S')}_{slug}_{secrets.token_hex(2)}"
        run_dir = date_dir / run_id
        try:
            run_dir.mkdir(parents=True)
            return run_dir, run_id
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique run directory under {date_dir}")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json atomically.

    The manifest is the immutable identity record of a Run: once created it
    is never overwritten (resume/fixed re-runs preserve the original
    run_id, creation time and lineage facts).
    """
    atomic_write_json(Path(run_dir) / "manifest.json", manifest)


def read_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    """Load the run's immutable manifest, if present."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def append_resume_event(
    run_dir: str | Path,
    *,
    checkpoint: str,
    command: str | None,
    resume_type: str,
    config_diffs: list[str],
) -> None:
    """Append one line to ``resume_events.jsonl`` (append-only lineage log).

    Records when and from which checkpoint a run was resumed, how the
    configuration drifted from the original run, and the launch command.
    """
    from datetime import datetime

    path = Path(run_dir) / "resume_events.jsonl"
    event = {
        "resumed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "resume_type": resume_type,
        "command": command,
        "config_diffs": config_diffs,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def update_status(run_dir: str | Path, **fields: Any) -> dict[str, Any]:
    """Merge fields into status.json with a fresh timestamp (atomic)."""
    status_path = Path(run_dir) / "status.json"
    payload: dict[str, Any] = {}
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload.update(fields)
    payload["last_update"] = _now().isoformat(timespec="seconds")
    atomic_write_json(status_path, payload)
    return payload


def git_fingerprint(repo_dir: str | Path = ".") -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                list(args),
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run("git", "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    porcelain = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(porcelain)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for module in ("torch", "torch_npu", "numpy", "pyarrow", "yaml", "PIL"):
        try:
            imported = __import__(module)
            versions[module] = str(getattr(imported, "__version__", "unknown"))
        except ImportError:
            versions[module] = None
    return versions


def collect_environment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot of the machine/toolchain context for the manifest."""
    environment: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": package_versions(),
        "git": git_fingerprint(),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "rank": int(os.environ.get("RANK", 0)),
    }
    if config is not None:
        device_cfg = config.get("device", {})
        environment["accelerator"] = device_cfg.get("accelerator")
        environment["distributed"] = bool(
            config.get("distributed", {}).get("enabled", False)
        )
    return environment


def append_run_index(runs_root: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON line to ``<runs_root>/index.jsonl``."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    index_path = runs_root / "index.jsonl"
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with index_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def read_run_index(runs_root: str | Path) -> list[dict[str, Any]]:
    index_path = Path(runs_root) / "index.jsonl"
    if not index_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def render_summary_md(
    *,
    run_id: str | None,
    run_dir: Path,
    state: str,
    started: str,
    finished: str,
    duration_seconds: float | None,
    config: dict[str, Any],
    summary_payload: dict[str, Any] | None,
    error_message: str | None = None,
) -> str:
    """One page human summary of a run."""
    experiment = config.get("experiment", {})
    train_cfg = config.get("train", {})
    decision = config.get("decision", {})
    lines: list[str] = []
    lines.append(f"# Run summary — {experiment.get('name', run_id or 'run')}")
    lines.append("")
    lines.append(f"- State: **{state}**")
    if run_id:
        lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Started: {started}")
    lines.append(f"- Finished: {finished}")
    if duration_seconds is not None:
        lines.append(f"- Duration: {duration_seconds / 60:.1f} min")
    if error_message:
        lines.append(f"- Error: `{error_message}`")
    lines.append("")
    lines.append("## Key hyperparameters")
    lines.append("")
    lines.append(f"- Decision threshold: `{decision.get('threshold')}`")
    lines.append(f"- Local batch size: `{train_cfg.get('local_batch_size')}`")
    lines.append(f"- Max steps: `{train_cfg.get('max_steps')}`")
    lines.append(
        f"- Epochs x steps/epoch: `{train_cfg.get('epochs')}` x "
        f"`{train_cfg.get('steps_per_epoch')}`"
    )
    lines.append(
        f"- Learning rate: `{config.get('optimizer', {}).get('learning_rate')}`"
    )
    lines.append(f"- Seed: `{experiment.get('seed')}`")
    lines.append(f"- Model factory: `{config.get('model', {}).get('factory')}`")
    lines.append(
        f"- Base checkpoint: `{config.get('model', {}).get('checkpoint_path')}`"
    )
    evaluation_cfg = config.get("evaluation", {})
    lines.append(
        "- Eval cadence: probe every "
        f"`{evaluation_cfg.get('train_probe_every_steps')}`, quick "
        f"validation every "
        f"`{evaluation_cfg.get('val_quick_every_steps')}`, full "
        f"validation every "
        f"`{evaluation_cfg.get('val_full_every_steps')}`"
    )
    lines.append(
        f"- Selection metric: `{evaluation_cfg.get('selection_metric', 'global_f1_at_decision_threshold')}`"
    )
    early_cfg = config.get("early_stopping") or {}
    if early_cfg.get("enabled"):
        lines.append(
            f"- Early stopping: monitor `{early_cfg.get('monitor')}`, "
            f"patience `{early_cfg.get('patience_evaluations')}`, "
            f"min_delta `{early_cfg.get('min_delta')}`"
        )
    if summary_payload:
        last = summary_payload.get("last_checkpoint_metrics") or {}
        best = (
            summary_payload.get("best_validation_metrics")
            or summary_payload.get("best_observed_dev_test_metrics")
            or {}
        )

        def fmt(metrics: dict[str, Any]) -> str:
            if not metrics:
                return "n/a"
            selection = metrics.get("selection_score")
            selection_text = (
                f"{selection:.4f}" if isinstance(selection, (int, float)) else "n/a"
            )
            global_f1 = metrics.get(
                "global_f1_at_decision_threshold",
                metrics.get("global_f1_tau099"),
            )
            global_text = (
                f"{global_f1:.4f}" if isinstance(global_f1, (int, float)) else "n/a"
            )
            worst = metrics.get(
                "worst_game_f1_at_decision_threshold",
                metrics.get("worst_game_f1_tau099"),
            )
            worst_text = f"{worst:.4f}" if isinstance(worst, (int, float)) else "n/a"
            return (
                f"step={metrics.get('checkpoint_step', '?')} "
                f"selection={selection_text} global_f1={global_text} "
                f"worst_game_f1={worst_text}"
            )

        lines.append("")
        lines.append("## Results (validation)")
        lines.append("")
        lines.append(f"- Last: {fmt(last)}")
        lines.append(f"- Best: {fmt(best)}")
        topk = summary_payload.get("topk_checkpoints") or []
        if topk:
            lines.append("")
            lines.append("### Top-K checkpoints")
            lines.append("")
            lines.append("| rank | step | value | file |")
            lines.append("|---|---|---|---|")
            for rank_index, entry in enumerate(topk, start=1):
                value = entry.get("value")
                value_text = (
                    f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
                )
                lines.append(
                    f"| {rank_index} | {entry.get('step', '?')} | "
                    f"{value_text} | `model_{entry.get('tag', '')}.pth` |"
                )
        counts = summary_payload.get("test_evaluation_counts") or {}
        lines.append(
            f"- Evaluations: quick={counts.get('quick', 0)}, "
            f"full={counts.get('full', 0)}, "
            f"train_probe={counts.get('train_probe', 0)}"
        )
        early = summary_payload.get("early_stopping") or {}
        if early.get("stop_reason"):
            lines.append(
                f"- Early stopped at step {early.get('stopped_at_step')}: "
                f"{early.get('stop_reason')} "
                f"(best step {early.get('best_step')})"
            )
        if summary_payload.get("restored_best"):
            lines.append("- Restored best-selection weights before finishing.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Metrics: `{run_dir / 'train_metrics.jsonl'}`")
    lines.append(f"- Evaluation history: `{run_dir / 'metrics' / 'evaluation.jsonl'}`")
    lines.append(f"- Reports: `{run_dir / 'reports'}`")
    lines.append(f"- Checkpoints: `{run_dir / 'checkpoints'}`")
    lines.append(f"- Machine summary: `{run_dir / 'training_summary.json'}`")
    lines.append(f"- Resolved config: `{run_dir / 'resolved_config.json'}`")
    lines.append(f"- Manifest: `{run_dir / 'manifest.json'}`")
    lines.append(f"- Overview page: `{run_dir / 'overview.html'}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# overview.html: self-contained human report with inline SVG curves
# ---------------------------------------------------------------------------

# Legacy single-series training charts (kept for export tooling).
_CHART_FIELDS = (
    ("loss", "Loss", "#dc2626"),
    ("interval_samples_per_second", "Throughput (samples/s)", "#2563eb"),
    ("data_wait_ratio", "Data wait ratio", "#d97706"),
    ("learning_rate", "Learning rate", "#059669"),
    ("grad_norm", "Gradient norm", "#7c3aed"),
)


def read_training_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(run_dir) / "train_metrics.jsonl"
    if not metrics_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_evaluation_history(run_dir: str | Path) -> list[dict[str, Any]]:
    """Unified evaluation time series: metrics/evaluation.jsonl."""
    history_path = Path(run_dir) / "metrics" / "evaluation.jsonl"
    if not history_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _history_series(
    history: list[dict[str, Any]],
    *,
    split: str,
    field: str,
    scope: str | None = None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    # Neutral field names first; fall back to legacy _tau099 names so old
    # run histories still render.
    fallback = field.replace("_at_decision_threshold", "_tau099")
    for row in history:
        if row.get("split") != split:
            continue
        if scope is not None and row.get("scope") != scope:
            continue
        value = row.get(field)
        if value is None and fallback != field:
            value = row.get(fallback)
        step = row.get("step")
        if isinstance(value, (int, float)) and isinstance(step, (int, float)):
            points.append((float(step), float(value)))
    return points


def _training_series(
    rows: list[dict[str, Any]], *fields: str
) -> list[tuple[float, float]]:
    """First available field wins (interval average before raw fallback)."""
    for field in fields:
        points = [
            (float(row["step"]), float(row[field]))
            for row in rows
            if "step" in row and field in row and isinstance(row[field], (int, float))
        ]
        if points:
            return points
    return []


def _svg_multi_chart(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    markers: list[dict[str, Any]] | None = None,
    width: int = 560,
    height: int = 150,
) -> str:
    """Inline SVG with one or more (label, color, points) series.

    ``markers`` are vertical reference lines: dicts with ``step``,
    ``label`` and optional ``color``.
    """
    markers = markers or []
    active = [(label, color, points) for label, color, points in series if points]
    total_points = sum(len(points) for _, _, points in active)
    if total_points < 2:
        return '<p class="nodata">not enough data</p>'
    xs = [x for _, _, points in active for x, _ in points]
    ys = [y for _, _, points in active for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-12)
    pad = 6.0

    def project(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - min_x) / span_x * (width - 2 * pad)
        py = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        return round(px, 2), round(py, 2)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
    ]
    for marker in markers:
        step = float(marker.get("step", 0))
        if not (min_x <= step <= max_x):
            continue
        mx, _ = project(step, min_y)
        color = marker.get("color", "#94a3b8")
        parts.append(
            f'<line x1="{mx}" y1="{pad}" x2="{mx}" y2="{height - pad}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{min(mx + 2, width - 130)}" y="{pad + 8}" '
            f'font-size="9" fill="{color}">{_html_escape(marker.get("label", ""))}</text>'
        )
    for _, color, points in active:
        polyline = " ".join(f"{px},{py}" for px, py in (project(*p) for p in points))
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
    legend_y = height - 3
    legend_x = pad
    for label, color, points in active:
        if not points:
            continue
        parts.append(
            f'<text x="{legend_x}" y="{legend_y}" font-size="9" '
            f'fill="{color}">{_html_escape(label)}</text>'
        )
        legend_x += 12 + 6.2 * len(label)
    parts.append(
        f'<text x="{pad}" y="12" font-size="10" fill="#64748b">max {max_y:.4g}</text>'
    )
    parts.append(
        f'<text x="{width - 150}" y="12" font-size="10" fill="#64748b">'
        f"steps {min_x:.4g}\u2013{max_x:.4g}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _svg_chart(
    rows: list[dict[str, Any]],
    field: str,
    color: str,
    width: int = 560,
    height: int = 150,
) -> str:
    points = [
        (float(row["step"]), float(row[field]))
        for row in rows
        if field in row and row[field] is not None
    ]
    return _svg_multi_chart([(field, color, points)], width=width, height=height)


def _evaluation_markers(
    history: list[dict[str, Any]],
    status_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Vertical markers: best selection / best loss / degradation / stop."""
    markers: list[dict[str, Any]] = []
    full_rows = [
        row
        for row in history
        if row.get("split") == "validation" and row.get("scope") == "full"
    ]
    scored = [
        row for row in full_rows if isinstance(row.get("selection_score"), (int, float))
    ]
    if scored:
        best = max(scored, key=lambda row: float(row["selection_score"]))
        markers.append(
            {
                "step": best["step"],
                "label": f"best selection @{int(best['step'])}",
                "color": "#059669",
            }
        )
        degraded = [
            row
            for row in scored
            if float(row["step"]) > float(best["step"])
            and float(row["selection_score"]) < float(best["selection_score"])
        ]
        if degraded:
            markers.append(
                {
                    "step": degraded[0]["step"],
                    "label": f"generalization degrades @{int(degraded[0]['step'])}",
                    "color": "#dc2626",
                }
            )
    loss_rows = [
        row for row in full_rows if isinstance(row.get("cross_entropy"), (int, float))
    ]
    if loss_rows:
        best_loss = min(loss_rows, key=lambda row: float(row["cross_entropy"]))
        markers.append(
            {
                "step": best_loss["step"],
                "label": f"best val loss @{int(best_loss['step'])}",
                "color": "#2563eb",
            }
        )
    status_payload = status_payload or {}
    early_step = status_payload.get("early_stopped_step")
    if isinstance(early_step, (int, float)):
        markers.append(
            {
                "step": early_step,
                "label": f"early stop @{int(early_step)}",
                "color": "#d97706",
            }
        )
    return markers


def _html_escape(text: str) -> str:
    import html

    return html.escape(str(text), quote=True)


def render_overview_html(
    *,
    run_dir: Path,
    run_id: str | None,
    state: str,
    started: str,
    finished: str | None,
    duration_seconds: float | None,
    config: dict[str, Any],
    summary_payload: dict[str, Any] | None,
) -> str:
    rows = read_training_metrics(run_dir)
    history = read_evaluation_history(run_dir)
    status_payload = None
    status_path = Path(run_dir) / "status.json"
    if status_path.is_file():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status_payload = None
    experiment = config.get("experiment", {})
    train_cfg = config.get("train", {})
    evaluation_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})
    early_stopping_cfg = config.get("early_stopping", {})
    state_color = {
        STATE_SUCCEEDED: "#059669",
        STATE_FAILED: "#dc2626",
    }.get(state, "#d97706")
    markers = _evaluation_markers(history, status_payload)
    probe_ce = _history_series(history, split="train_probe", field="cross_entropy")
    val_ce = _history_series(
        history, split="validation", field="cross_entropy", scope="full"
    )
    probe_selection = _history_series(
        history, split="train_probe", field="selection_score"
    )
    val_selection = _history_series(
        history, split="validation", field="selection_score", scope="full"
    )
    chart_cards = []
    chart_cards.append(
        '<div class="card"><h3>Training Losses (interval averages)</h3>'
        + _svg_multi_chart(
            [
                (
                    "total loss",
                    "#dc2626",
                    _training_series(rows, "interval_loss", "loss"),
                ),
                (
                    "cross entropy",
                    "#2563eb",
                    _training_series(rows, "interval_ce", "ce"),
                ),
                (
                    "threshold loss",
                    "#d97706",
                    _training_series(rows, "interval_threshold_loss", "threshold_loss"),
                ),
            ]
        )
        + "</div>"
    )
    chart_cards.append(
        '<div class="card"><h3>Generalization: cross entropy '
        "(train probe vs validation)</h3>"
        + _svg_multi_chart(
            [
                ("train probe CE", "#dc2626", probe_ce),
                ("validation CE", "#059669", val_ce),
            ],
            markers,
        )
        + "</div>"
    )
    chart_cards.append(
        '<div class="card"><h3>Generalization: selection score '
        "(train probe vs validation)</h3>"
        + _svg_multi_chart(
            [
                ("train probe selection", "#dc2626", probe_selection),
                ("validation selection", "#059669", val_selection),
            ],
            markers,
        )
        + "</div>"
    )
    chart_cards.append(
        '<div class="card"><h3>Validation F1 (global / macro-game / '
        "worst-game)</h3>"
        + _svg_multi_chart(
            [
                (
                    "global F1",
                    "#059669",
                    _history_series(
                        history,
                        split="validation",
                        field="global_f1_at_decision_threshold",
                        scope="full",
                    ),
                ),
                (
                    "macro-game F1",
                    "#2563eb",
                    _history_series(
                        history,
                        split="validation",
                        field="macro_game_f1_at_decision_threshold",
                        scope="full",
                    ),
                ),
                (
                    "worst-game F1",
                    "#dc2626",
                    _history_series(
                        history,
                        split="validation",
                        field="worst_game_f1_at_decision_threshold",
                        scope="full",
                    ),
                ),
            ],
            markers,
        )
        + "</div>"
    )
    chart_cards.append(
        '<div class="card"><h3>Validation calibration (Brier / ECE)</h3>'
        + _svg_multi_chart(
            [
                (
                    "Brier",
                    "#7c3aed",
                    _history_series(
                        history,
                        split="validation",
                        field="brier_score",
                        scope="full",
                    ),
                ),
                (
                    "ECE (20 bins)",
                    "#d97706",
                    _history_series(
                        history,
                        split="validation",
                        field="ece_20_bins",
                        scope="full",
                    ),
                ),
            ],
            markers,
        )
        + "</div>"
    )
    for field, title, color in (
        ("learning_rate", "Learning rate", "#059669"),
        ("grad_norm", "Gradient norm", "#7c3aed"),
        ("interval_samples_per_second", "Throughput (samples/s)", "#2563eb"),
        ("data_wait_ratio", "Data wait ratio", "#d97706"),
    ):
        chart_cards.append(
            f'<div class="card"><h3>{title}</h3>{_svg_chart(rows, field, color)}</div>'
        )
    charts = "\n".join(chart_cards)
    split_migration = data_cfg.get("split_migration") or {}
    split_roles = (
        "train / validation / test"
        if not split_migration.get("test_used_as_validation")
        else "train / validation (test aliased as validation \u2014 NO "
        "independent test set)"
    )
    hyperparameters = (
        ("Decision threshold", config.get("decision", {}).get("threshold")),
        ("Split roles", split_roles),
        ("Local batch size", train_cfg.get("local_batch_size")),
        ("Max steps", train_cfg.get("max_steps")),
        (
            "Epochs x steps/epoch",
            f"{train_cfg.get('epochs')} x {train_cfg.get('steps_per_epoch')}",
        ),
        ("Learning rate", config.get("optimizer", {}).get("learning_rate")),
        ("Seed", experiment.get("seed")),
        ("Model factory", config.get("model", {}).get("factory")),
        ("Base checkpoint", config.get("model", {}).get("checkpoint_path")),
        ("Data backend", data_cfg.get("backend", "png")),
        ("Train probe cadence", evaluation_cfg.get("train_probe_every_steps")),
        ("Quick validation cadence", evaluation_cfg.get("val_quick_every_steps")),
        ("Full validation cadence", evaluation_cfg.get("val_full_every_steps")),
        (
            "Selection metric",
            evaluation_cfg.get("selection_metric", "global_f1_at_decision_threshold"),
        ),
        (
            "Early stopping",
            (
                f"monitor={early_stopping_cfg.get('monitor')} "
                f"patience={early_stopping_cfg.get('patience_evaluations')} "
                f"min_delta={early_stopping_cfg.get('min_delta')}"
            )
            if early_stopping_cfg.get("enabled")
            else "disabled",
        ),
    )
    hyper_rows = "\n".join(
        f"<tr><td>{_html_escape(label)}</td>"
        f"<td><code>{_html_escape(value)}</code></td></tr>"
        for label, value in hyperparameters
    )
    metrics_rows = ""
    if summary_payload:
        for label, key in (
            ("Last full validation", "last_checkpoint_metrics"),
            ("Best validation", "best_validation_metrics"),
            ("Best validation (legacy key)", "best_observed_dev_test_metrics"),
        ):
            metrics = summary_payload.get(key) or {}
            if not metrics:
                continue

            def cell(name: str, metrics=metrics) -> str:
                value = metrics.get(name)
                if isinstance(value, (int, float)):
                    return f"{value:.4f}"
                return str(value if value is not None else "n/a")

            metrics_rows += (
                f"<tr><td>{label}</td><td>{cell('checkpoint_step')}</td>"
                f"<td>{cell('selection_score')}</td>"
                f"<td>{cell('global_f1_at_decision_threshold')}</td>"
                f"<td>{cell('macro_game_f1_at_decision_threshold')}</td>"
                f"<td>{cell('worst_game_f1_at_decision_threshold')}</td></tr>"
            )
    duration_text = f"{duration_seconds / 60:.1f} min" if duration_seconds else "n/a"
    finished_text = finished or "running"
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Run {_html_escape(run_id or experiment.get("name", "run"))}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; color: #0f172a; }}
  h1 {{ font-size: 1.4rem; }} h3 {{ margin: 0 0 .5rem 0; font-size: .95rem; }}
  .banner {{ display:inline-block; padding:.25rem .75rem; border-radius:.5rem;
             color:white; background:{state_color}; font-weight:600; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(560px,1fr)); gap:1rem; }}
  .card {{ border:1px solid #e2e8f0; border-radius:.5rem; padding:1rem; }}
  table {{ border-collapse: collapse; width:100%; font-size:.9rem; }}
  td, th {{ border:1px solid #e2e8f0; padding:.35rem .6rem; text-align:left; }}
  th {{ background:#f1f5f9; }}
  code {{ background:#f1f5f9; padding:.1rem .3rem; border-radius:.25rem; }}
  .nodata {{ color:#94a3b8; font-size:.85rem; }}
  svg {{ width:100%; height:auto; }}
</style>
</head>
<body>
<h1>Run {_html_escape(run_id or experiment.get("name", "run"))}</h1>
<p><span class="banner">{_html_escape(state)}</span>
   &nbsp;{_html_escape(started)} \u2192 {_html_escape(finished_text)}
   ({_html_escape(duration_text)})</p>
<div class="grid">
  <div class="card"><h3>Key hyperparameters</h3>
    <table>{hyper_rows}</table></div>
  <div class="card"><h3>Evaluation results (validation)</h3>
    <table><tr><th>source</th><th>step</th><th>selection</th>
    <th>global F1</th><th>macro-game F1</th><th>worst-game F1</th></tr>
    {metrics_rows or '<tr><td colspan=6 class="nodata">no evaluation recorded</td></tr>'}
    </table></div>
</div>
<h2>Training &amp; generalization curves</h2>
<div class="grid">{charts}</div>
<p style="color:#64748b;font-size:.85rem">Self-contained report generated by
CLSTrainer. Machine-readable data: train_metrics.jsonl,
metrics/evaluation.jsonl, training_summary.json, reports/.</p>
</body>
</html>
"""
