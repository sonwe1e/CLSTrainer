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
    raise RuntimeError(
        f"Could not allocate a unique run directory under {date_dir}"
    )


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2)
    )


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(Path(run_dir) / "manifest.json", manifest)


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
    lines.append(
        f"- Model factory: `{config.get('model', {}).get('factory')}`"
    )
    lines.append(
        f"- Base checkpoint: `{config.get('model', {}).get('checkpoint_path')}`"
    )
    evaluation_cfg = config.get("evaluation", {})
    lines.append(
        "- Eval cadence: quick every "
        f"`{evaluation_cfg.get('quick_test_every_steps')}`, full every "
        f"`{evaluation_cfg.get('full_test_every_steps')}`"
    )
    lines.append(
        f"- Selection metric: `{evaluation_cfg.get('selection_metric', 'global_f1_tau099')}`"
    )
    if summary_payload:
        last = summary_payload.get("last_checkpoint_metrics") or {}
        best = summary_payload.get("best_observed_dev_test_metrics") or {}

        def fmt(metrics: dict[str, Any]) -> str:
            if not metrics:
                return "n/a"
            selection = metrics.get("selection_score")
            selection_text = (
                f"{selection:.4f}" if isinstance(selection, (int, float)) else "n/a"
            )
            global_f1 = metrics.get("global_f1_tau099")
            global_text = (
                f"{global_f1:.4f}" if isinstance(global_f1, (int, float)) else "n/a"
            )
            worst = metrics.get("worst_game_f1_tau099")
            worst_text = (
                f"{worst:.4f}" if isinstance(worst, (int, float)) else "n/a"
            )
            return (
                f"step={metrics.get('checkpoint_step', '?')} "
                f"selection={selection_text} global_f1={global_text} "
                f"worst_game_f1={worst_text}"
            )

        lines.append("")
        lines.append("## Results (observed dev-test)")
        lines.append("")
        lines.append(f"- Last: {fmt(last)}")
        lines.append(f"- Best: {fmt(best)}")
        counts = summary_payload.get("test_evaluation_counts") or {}
        lines.append(
            f"- Evaluations: quick={counts.get('quick', 0)}, "
            f"full={counts.get('full', 0)}"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Metrics: `{run_dir / 'train_metrics.jsonl'}`")
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
    if len(points) < 2:
        return '<p class="nodata">not enough data</p>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-12)
    pad = 6.0

    def project(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - min_x) / span_x * (width - 2 * pad)
        py = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        return round(px, 2), round(py, 2)

    polyline = " ".join(f"{px},{py}" for px, py in (project(*p) for p in points))
    def fmt(value: float) -> str:
        return f"{value:.4g}"

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        'stroke-width="2"/>'
        f'<text x="{pad}" y="12" font-size="10" fill="#64748b">'
        f'max {fmt(max_y)}</text>'
        f'<text x="{pad}" y="{height - 3}" font-size="10" fill="#64748b">'
        f'min {fmt(min_y)}</text>'
        f'<text x="{width - 90}" y="{height - 3}" font-size="10" '
        f'fill="#64748b">steps {fmt(min_x)}\u2013{fmt(max_x)}</text>'
        '</svg>'
    )


def _html_escape(text: str) -> str:
    import html

    return html.escape(str(text), quote=True)


def render_overview_html(
    *,
    run_dir: Path,
    run_id: str | None,
    state: str,
    started: str,
    finished: str,
    duration_seconds: float | None,
    config: dict[str, Any],
    summary_payload: dict[str, Any] | None,
) -> str:
    rows = read_training_metrics(run_dir)
    experiment = config.get("experiment", {})
    train_cfg = config.get("train", {})
    evaluation_cfg = config.get("evaluation", {})
    state_color = {
        STATE_SUCCEEDED: "#059669",
        STATE_FAILED: "#dc2626",
    }.get(state, "#d97706")
    charts = "\n".join(
        f'<div class="card"><h3>{title}</h3>'
        f'{_svg_chart(rows, field, color)}</div>'
        for field, title, color in _CHART_FIELDS
    )
    hyperparameters = (
        ("Decision threshold", config.get("decision", {}).get("threshold")),
        ("Local batch size", train_cfg.get("local_batch_size")),
        ("Max steps", train_cfg.get("max_steps")),
        ("Epochs x steps/epoch", f"{train_cfg.get('epochs')} x {train_cfg.get('steps_per_epoch')}"),
        ("Learning rate", config.get("optimizer", {}).get("learning_rate")),
        ("Seed", experiment.get("seed")),
        ("Model factory", config.get("model", {}).get("factory")),
        ("Base checkpoint", config.get("model", {}).get("checkpoint_path")),
        ("Data backend", config.get("data", {}).get("backend", "png")),
        ("Quick test cadence", evaluation_cfg.get("quick_test_every_steps")),
        ("Full test cadence", evaluation_cfg.get("full_test_every_steps")),
        ("Selection metric", evaluation_cfg.get("selection_metric", "global_f1_tau099")),
    )
    hyper_rows = "\n".join(
        f"<tr><td>{_html_escape(label)}</td>"
        f"<td><code>{_html_escape(value)}</code></td></tr>"
        for label, value in hyperparameters
    )
    metrics_rows = ""
    if summary_payload:
        for label, key in (
            ("Last checkpoint", "last_checkpoint_metrics"),
            ("Best observed dev-test", "best_observed_dev_test_metrics"),
        ):
            metrics = summary_payload.get(key) or {}
            def cell(name: str) -> str:
                value = metrics.get(name)
                if isinstance(value, (int, float)):
                    return f"{value:.4f}"
                return str(value if value is not None else "n/a")
            metrics_rows += (
                f"<tr><td>{label}</td><td>{cell('checkpoint_step')}</td>"
                f"<td>{cell('selection_score')}</td>"
                f"<td>{cell('global_f1_tau099')}</td>"
                f"<td>{cell('macro_game_f1_tau099')}</td>"
                f"<td>{cell('worst_game_f1_tau099')}</td></tr>"
            )
    duration_text = (
        f"{duration_seconds / 60:.1f} min" if duration_seconds else "n/a"
    )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Run {_html_escape(run_id or experiment.get('name', 'run'))}</title>
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
<h1>Run {_html_escape(run_id or experiment.get('name', 'run'))}</h1>
<p><span class="banner">{_html_escape(state)}</span>
   &nbsp;{_html_escape(started)} \u2192 {_html_escape(finished)}
   ({_html_escape(duration_text)})</p>
<div class="grid">
  <div class="card"><h3>Key hyperparameters</h3>
    <table>{hyper_rows}</table></div>
  <div class="card"><h3>Evaluation results (observed dev-test)</h3>
    <table><tr><th>source</th><th>step</th><th>selection</th>
    <th>global F1</th><th>macro-game F1</th><th>worst-game F1</th></tr>
    {metrics_rows or '<tr><td colspan=6 class="nodata">no evaluation recorded</td></tr>'}
    </table></div>
</div>
<h2>Training curves</h2>
<div class="grid">{charts}</div>
<p style="color:#64748b;font-size:.85rem">Self-contained report generated by
CLSTrainer. Machine-readable data: train_metrics.jsonl, training_summary.json,
reports/.</p>
</body>
</html>
"""
