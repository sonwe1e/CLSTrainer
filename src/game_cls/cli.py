"""cls-trainer command line interface.

Workflows:

    cls-trainer train --config configs/npu_1p.yaml [key=value ...]
    cls-trainer train --config ... --dry-run
    cls-trainer train --resume <run_dir>
    cls-trainer config show --config ... [--with-source]
    cls-trainer config validate --config ...
    cls-trainer config reference
    cls-trainer run list [--root runs]
    cls-trainer run show latest|<run_dir>
    cls-trainer doctor --config ...

Every ``train`` start defaults to ``--run-mode unique``: the configured
``experiment.output_dir`` is treated as a runs root and a fresh timestamped
run directory is allocated, so re-running a command can never overwrite a
previous run. ``--run-mode fixed`` restores the legacy in-place behavior.
"""

from __future__ import annotations

import argparse
import json
import os
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
        self._file = (Path(run_dir) / "console.log").open(
            "a", encoding="utf-8"
        )
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
    "experiment.run_mode",
    "train.resume_path",
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
    "data.class_probability",
    "data.delta_probability",
    "data.game_alpha",
    "model.factory",
    "model.trainable_name_contains",
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


def _flatten_dict(
    node: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
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
        warning.split(":", 1)[0].strip() in RESUME_EXTEND_KEYS
        for warning in warnings
    )
    return "extend" if flexible_only else "fork"


def _resolve_run_dir(target: str, root: Path) -> Path:
    if target == "latest":
        records = _find_index_records(root)
        if records:
            return Path(records[-1]["output_dir"])
        candidates = sorted(
            path
            for path in root.glob("*/*")
            if (path / "status.json").is_file()
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


def _dry_run_report(config: dict[str, Any], config_file: str) -> int:
    train_cfg = config["train"]
    evaluation_cfg = config["evaluation"]
    device_cfg = config["device"]
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_batch = int(train_cfg["local_batch_size"])
    max_steps = train_cfg.get("max_steps")
    total_steps = int(
        max_steps or int(train_cfg["epochs"]) * int(train_cfg["steps_per_epoch"])
    )
    stop_after = train_cfg.get("stop_after_steps")
    checkpoint_path = config["model"].get("checkpoint_path")
    checkpoint_status = "not configured"
    if checkpoint_path:
        checkpoint_status = (
            "exists" if Path(checkpoint_path).is_file() else "MISSING"
        )
    data_cfg = config["data"]
    warnings: list[str] = []
    if data_cfg.get("synthetic"):
        warnings.append("data.synthetic=true — smoke-test data only.")
    if not config["model"].get("factory", "").strip():
        warnings.append("model.factory is empty.")
    if "your_package" in str(config["model"].get("factory", "")):
        warnings.append("model.factory is still the placeholder.")
    if checkpoint_path and checkpoint_status == "MISSING":
        warnings.append(f"model.checkpoint_path does not exist: {checkpoint_path}")
    quick_every = int(
        evaluation_cfg.get(
            "val_quick_every_steps",
            evaluation_cfg.get("quick_test_every_steps", 0),
        )
    )
    full_every = int(
        evaluation_cfg.get(
            "val_full_every_steps",
            evaluation_cfg.get("full_test_every_steps", 0),
        )
    )
    probe_every = int(evaluation_cfg.get("train_probe_every_steps", 0))

    print("=== DRY RUN — nothing will be initialized or written ===")
    print(f"config file        : {config_file}")
    print(f"accelerator        : {device_cfg.get('accelerator')}")
    print(f"world size         : {world_size} (from environment)")
    print(f"model factory      : {config['model'].get('factory')}")
    print(
        f"base checkpoint    : {checkpoint_path or 'none'} "
        f"({checkpoint_status})"
    )
    print(
        "data backend       : "
        f"{data_cfg.get('backend', 'png')} "
        f"(synthetic={bool(data_cfg.get('synthetic', False))})"
    )
    print(f"decision threshold : {config['decision']['threshold']}")
    print(f"local batch size   : {local_batch}")
    print(f"global batch size  : {local_batch * world_size}")
    print(f"total steps        : {total_steps}")
    if stop_after is not None:
        print(f"stop after         : {stop_after} steps")
    print(
        f"train probe        : "
        f"{'every ' + str(probe_every) + ' steps' if probe_every else 'disabled'}"
    )
    print(
        f"quick validation   : "
        f"{'every ' + str(quick_every) + ' steps' if quick_every else 'disabled'}"
    )
    print(
        f"full validation    : "
        f"{'every ' + str(full_every) + ' steps' if full_every else 'disabled'}"
        f" (at end: {bool(evaluation_cfg.get('val_full_at_end', evaluation_cfg.get('full_test_at_end', True)))})"
    )
    data_cfg = config["data"]
    if data_cfg.get("synthetic"):
        print("split roles        : synthetic (train/val/test)")
    elif (data_cfg.get("split_migration") or {}).get(
        "test_used_as_validation"
    ):
        print(
            "split roles        : test aliased as validation — NO "
            "independent test set"
        )
    else:
        print("split roles        : train / validation / test")
    early_cfg = config.get("early_stopping") or {}
    if early_cfg.get("enabled"):
        print(
            f"early stopping     : monitor={early_cfg.get('monitor')} "
            f"patience={early_cfg.get('patience_evaluations')} "
            f"min_delta={early_cfg.get('min_delta')} "
            f"burn_in={early_cfg.get('burn_in_steps')}"
        )
    else:
        print("early stopping     : disabled")
    run_mode = str(config["experiment"].get("run_mode", "fixed"))
    print(f"run mode           : {run_mode}")
    print(f"runs root          : {config['experiment']['output_dir']}")
    if run_mode == "unique":
        print(
            "planned output     : <runs root>/<date>/<time>_<name>_<id>/ "
            "(allocated at start)"
        )
    else:
        print(
            f"planned output     : {config['experiment']['output_dir']} "
            "(written in place)"
        )
    print(f"config warnings    : {len(warnings)}")
    for warning in warnings:
        print(f"  [WARNING] {warning}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.engine.trainer import run_training
    from game_cls.runs import collect_environment, sha256_file

    try:
        if args.resume and args.fork:
            raise SystemExit("--resume and --fork are mutually exclusive.")
        if args.resume:
            resume_dir = Path(args.resume).resolve()
            if not resume_dir.is_dir():
                raise SystemExit(
                    f"Resume run directory does not exist: {resume_dir}"
                )
            config_source = args.config or str(
                resume_dir / "resolved_config.json"
            )
            if not Path(config_source).is_file():
                raise SystemExit(
                    f"No config found for resume: pass --config or ensure "
                    f"{resume_dir / 'resolved_config.json'} exists."
                )
        elif args.fork:
            fork_dir = _resolve_run_dir(
                args.fork, Path(args.runs_root)
            )
            config_source = args.config or str(
                fork_dir / "resolved_config.json"
            )
            if not Path(config_source).is_file():
                raise SystemExit(
                    f"No config found for fork: pass --config or ensure "
                    f"{fork_dir / 'resolved_config.json'} exists."
                )
        else:
            if not args.config:
                raise SystemExit("train requires --config (or --resume/--fork).")
            config_source = args.config
        config = load_config(config_source, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2

    parent_run_id: str | None = None
    forked_from: str | None = None
    resume_run_id: str | None = None
    inferred_runs_root: str | None = None
    resume_type = "exact"
    resume_config_diffs: list[str] = []
    if args.resume:
        baseline = _read_run_json(resume_dir, "resolved_config.json")
        if baseline:
            critical, drift_warnings = check_resume_drift(baseline, config)
            for warning in drift_warnings:
                print(f"[WARNING] resume config drift: {warning}")
            if critical:
                print(
                    "Resume refused: critical config drift detected:",
                    file=sys.stderr,
                )
                for problem in critical:
                    print(f"  - {problem}", file=sys.stderr)
                print(
                    "These fields change what the restored state means. "
                    "Start a new run (optionally with --fork) instead.",
                    file=sys.stderr,
                )
                return 3
            resume_config_diffs = drift_warnings
            resume_type = classify_resume(baseline, config, drift_warnings)
            if resume_type == "fork":
                print(
                    "Resume refused: this would change the training "
                    "strategy, which invalidates the restored state. "
                    "Use --fork RUN_ID to start a new run derived from "
                    "this one instead.",
                    file=sys.stderr,
                )
                return 3
            if resume_type == "extend":
                print(
                    "[WARNING] resume-extend: max_steps/stop_after changed; "
                    "the scheduler is re-planned for the new budget."
                )
        resume_manifest = _read_run_json(resume_dir, "manifest.json") or {}
        resume_run_id = resume_manifest.get("run_id")
        # Unique-mode runs live at <runs_root>/<YYYYMMDD>/<run_id>; recover
        # the original runs root so index/status updates land next to the
        # run's siblings instead of inside the run directory.
        if resume_run_id:
            candidate = resume_dir.parent.parent
            if candidate.is_dir():
                inferred_runs_root = str(candidate)
        config["experiment"]["output_dir"] = str(resume_dir)
        config["experiment"]["run_mode"] = "fixed"
        config["train"]["resume_path"] = _resolve_resume_checkpoint(
            resume_dir, config
        )
    elif args.fork:
        fork_manifest = _read_run_json(fork_dir, "manifest.json") or {}
        parent_run_id = fork_manifest.get("run_id")
        forked_from = str(fork_dir)
        explicit_resume = any(
            override.split("=", 1)[0] == "train.resume_path"
            for override in args.overrides
        )
        if not explicit_resume:
            config["train"]["resume_path"] = None
        config["experiment"]["run_mode"] = "unique"
    else:
        config["experiment"]["run_mode"] = args.run_mode

    if args.dry_run:
        return _dry_run_report(config, config_source)

    _maybe_prepare_split(config)

    checkpoint_path = config["model"].get("checkpoint_path")
    checkpoint_hash = None
    rank = int(os.environ.get("RANK", 0))
    if checkpoint_path and Path(checkpoint_path).is_file() and rank == 0:
        if os.environ.get("CLS_SKIP_CHECKPOINT_HASH"):
            checkpoint_hash = "skipped"
        else:
            checkpoint_hash = sha256_file(checkpoint_path)

    run_meta = {
        "command": " ".join(sys.argv),
        "config_file": config_source,
        "resumed_from": args.resume,
        "forked_from": forked_from,
        "parent_run_id": parent_run_id,
        "run_id": resume_run_id,
        "resume_type": resume_type,
        "resume_config_diffs": resume_config_diffs,
        "runs_root": inferred_runs_root
        or str(Path(args.runs_root).resolve()),
        "base_checkpoint_sha256": checkpoint_hash,
        "environment": collect_environment(config),
    }

    with _TeeContext() as tee:
        result = run_training(
            config, run_meta=run_meta, on_run_dir=tee.attach
        )

    output_dir = result.get("output_dir", "?")
    run_dir = Path(output_dir)
    print("")
    print("=== Training finished ===")
    print(f"state        : {result.get('state')}")
    print(f"run directory: {output_dir}")
    print(f"final step   : {result.get('global_step')}")
    best = result.get("best_metrics") or {}
    if isinstance(best, dict) and best:
        score = best.get("selection_score")
        if isinstance(score, (int, float)):
            print(f"best selection score: {score:.4f}")
    print("")
    print("Where to look:")
    print(f"  human summary     : {run_dir / 'summary.md'}")
    print(f"  status            : {run_dir / 'status.json'}")
    print(f"  manifest          : {run_dir / 'manifest.json'}")
    print(f"  training curves   : {run_dir / 'train_metrics.jsonl'}")
    print(f"  evaluation reports: {run_dir / 'reports'}")
    print(f"  checkpoints       : {run_dir / 'checkpoints'}")
    return 0


RECIPE_TEMPLATE = """\
# Recipe generated by: cls-trainer init --profile {profile}
# Fill the REPLACE_ME fields, then:
#
#   cls-trainer doctor --config {output}
#   cls-trainer train  --config {output} --dry-run
#
# Business facts (threshold 0.99, delta=2 test pairs, cls-only training,
# 448x208 frames) come from the task profile and are intentionally absent here.
# Device/DataLoader stability rules come from the profile.
profile: {profile}

presets:
  augmentation: standard
  dataloader: stable
  evaluation: production

experiment:
  name: {name}
  seed: 20260728
  output_dir: runs/{name}

model:
  factory: REPLACE_ME_PACKAGE.module:build_model
  checkpoint_path: REPLACE_ME/path/to/base_model.pt

data:
  synthetic: false
  require_content_hash_audit: true
  require_unique_video_keys_across_splits: true
  require_independent_test: true
  audit_path: indexes/audit.json
  train_video_index: indexes/train_video_entries.parquet
  val_video_index: indexes/val_video_entries.parquet
  test_video_index: indexes/test_video_entries.parquet
  train_index: indexes/train_frames.parquet
  val_index: indexes/val_frames.parquet
  test_index: indexes/test_frames.parquet
  backend: png
  duplicate_policy:
    same_label_cross_split: error
    same_label_within_split: warning
    cross_label_same_content: error
    same_basename: info

pair:
  train_delta_probability: {{1: 0.15, 2: 0.70, 3: 0.15}}

sampler:
  game_alpha: 0.25
  class_probability: {{0: 0.5, 1: 0.5}}
  deduplicate_within_global_batch: true

optimizer:
  learning_rate: 0.001
  weight_decay: 0.0001

scheduler:
  warmup_steps: 500
  min_learning_rate: 0.00001

train:
  epochs: 10
  steps_per_epoch: 1000
  max_steps: null
  stop_after_steps: null
  resume_path: null
  verify_frozen_parameters: false
  local_batch_size: 64
  gradient_clip_norm: 5.0
  log_every_steps: 50

checkpoint:
  save_last_every_steps: 1000
  save_best_selection: true
  save_best_val_loss: true
  save_best_worst_game: true

# Early stopping watches full validation only; max_steps is a safety cap.
early_stopping:
  enabled: true
  monitor: selection_score
  mode: max
  full_validation_only: true
  burn_in_steps: 6000
  patience_evaluations: 3
  min_delta: 0.001
  restore_best: true
"""


def cmd_init(args: argparse.Namespace) -> int:
    from game_cls.config import list_available_layers

    layers = list_available_layers()
    if args.profile not in layers["profiles"]:
        print(
            f"Unknown profile '{args.profile}'. Available profiles: "
            f"{', '.join(layers['profiles']) or '(none found under configs/profiles/)'}",
            file=sys.stderr,
        )
        return 2
    output = Path(args.output)
    if output.exists() and not args.force:
        print(
            f"{output} already exists; pass --force to overwrite.",
            file=sys.stderr,
        )
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        RECIPE_TEMPLATE.format(
            profile=args.profile, name=args.name, output=str(output)
        ),
        encoding="utf-8",
    )
    print(f"Recipe written to {output}")
    print("Next steps:")
    print(f"  1. Replace the REPLACE_ME fields in {output}")
    print(f"  2. cls-trainer doctor --config {output}")
    print(f"  3. cls-trainer train  --config {output} --dry-run")
    return 0


# ---------------------------------------------------------------------------
# config show / validate / reference
# ---------------------------------------------------------------------------


def cmd_config_show(args: argparse.Namespace) -> int:
    from game_cls.config import load_config, load_config_with_sources

    if args.with_source:
        config, sources = load_config_with_sources(args.config, args.overrides)
        from game_cls.config_schema import describe_reference

        order = {row["path"]: i for i, row in enumerate(describe_reference())}

        def flatten(node: dict, prefix: str = "") -> list[tuple[str, Any]]:
            rows: list[tuple[str, Any]] = []
            for key, value in node.items():
                dotted = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    rows.extend(flatten(value, dotted))
                else:
                    rows.append((dotted, value))
            return rows

        known = {dotted for dotted, _ in flatten(config)}
        flattened = flatten(config)
        flattened.sort(key=lambda item: order.get(item[0], 10_000))
        for dotted, value in flattened:
            origin = sources.get(dotted, "<default>")
            print(f"{dotted} = {value!r}    # from {origin}")
        extra_sources = {
            dotted: origin
            for dotted, origin in sources.items()
            if dotted not in known
        }
        for dotted, origin in sorted(extra_sources.items()):
            print(f"{dotted} = <not set>    # from {origin}")
        return 0

    config = load_config(args.config, args.overrides)
    try:
        import yaml

        print(yaml.safe_dump(config, sort_keys=True, allow_unicode=True))
    except ImportError:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        print("Config INVALID:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    train_cfg = config["train"]
    total_steps = int(
        train_cfg.get("max_steps")
        or int(train_cfg["epochs"]) * int(train_cfg["steps_per_epoch"])
    )
    print(f"Config OK: {args.config}")
    print(f"  decision threshold: {config['decision']['threshold']}")
    print(f"  total steps       : {total_steps}")
    print(f"  local batch size  : {train_cfg['local_batch_size']}")
    print(f"  output dir        : {config['experiment']['output_dir']}")
    return 0


def cmd_config_reference(args: argparse.Namespace) -> int:
    from game_cls.config_schema import describe_reference

    for row in describe_reference():
        flags = []
        if row["legacy"]:
            flags.append("legacy")
        if row["choices"]:
            flags.append("choices: " + "|".join(map(str, row["choices"])))
        suffix = f" ({'; '.join(flags)})" if flags else ""
        print(f"{row['path']}  [{row['type']}]{suffix}")
        print(f"    {row['description']}")
    return 0


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


def cmd_run_list(args: argparse.Namespace) -> int:
    root = Path(args.root)
    records = _find_index_records(root)
    if not records:
        print(f"No runs recorded under {root.resolve()} yet.")
        return 0
    header = f"{'RUN ID':42s} {'STATE':10s} {'STEPS':>10s} FINISHED"
    print(header)
    print("-" * len(header))
    for record in records[-args.limit :]:
        print(
            f"{str(record.get('run_id') or '?'):42s} "
            f"{str(record.get('state') or '?'):10s} "
            f"{str(record.get('global_step') or '-'):>10s} "
            f"{record.get('finished', '?')}"
        )
        print(f"    {record.get('output_dir')}")
    return 0


def cmd_run_show(args: argparse.Namespace) -> int:
    root = Path(args.root)
    run_dir = _resolve_run_dir(args.target, root)
    print(f"run directory: {run_dir}")

    manifest = _read_run_json(run_dir, "manifest.json")
    status = _read_run_json(run_dir, "status.json")
    summary = _read_run_json(run_dir, "training_summary.json")
    if manifest:
        print(f"name         : {manifest.get('run_name')}")
        print(f"run id       : {manifest.get('run_id')}")
        print(f"created      : {manifest.get('created')}")
        print(f"command      : {manifest.get('command')}")
        environment = manifest.get("environment") or {}
        git = environment.get("git") or {}
        print(
            f"git          : {git.get('commit')} "
            f"(dirty={git.get('dirty')})"
        )
        print(f"host         : {environment.get('hostname')}")
        print(
            f"accelerator  : {manifest.get('accelerator')} "
            f"world_size={manifest.get('world_size')}"
        )
        if manifest.get("parent_run_id"):
            print(
                f"parent run   : {manifest.get('parent_run_id')} "
                f"(forked from {manifest.get('forked_from')})"
            )
    if status:
        print(f"state        : {status.get('state')}")
        print(f"last update  : {status.get('last_update')}")
        if status.get("error_type"):
            print(f"error        : {status['error_type']}: {status.get('error_message')}")
    if summary:
        best = summary.get("best_observed_dev_test_metrics") or {}
        if isinstance(best.get("selection_score"), (int, float)):
            print(f"best score   : {best['selection_score']:.4f}")
        topk = summary.get("topk_checkpoints") or []
        if topk:
            print("topk         :")
            for entry in topk:
                value = entry.get("value")
                value_text = (
                    f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
                )
                print(
                    f"  step={entry.get('step', '?')} "
                    f"{entry.get('monitor', 'selection_score')}={value_text} "
                    f"model_{entry.get('tag', '')}.pth"
                )
    for artifact in ("summary.md", "overview.html", "train_metrics.jsonl", "console.log"):
        if (run_dir / artifact).is_file():
            print(f"artifact     : {run_dir / artifact}")
    return 0


# ---------------------------------------------------------------------------
# run compare / export-tensorboard
# ---------------------------------------------------------------------------


def cmd_run_compare(args: argparse.Namespace) -> int:
    root = Path(args.root)
    dir_a = _resolve_run_dir(args.run_a, root)
    dir_b = _resolve_run_dir(args.run_b, root)
    config_a = _read_run_json(dir_a, "resolved_config.json") or {}
    config_b = _read_run_json(dir_b, "resolved_config.json") or {}
    status_a = _read_run_json(dir_a, "status.json") or {}
    status_b = _read_run_json(dir_b, "status.json") or {}
    summary_a = _read_run_json(dir_a, "training_summary.json") or {}
    summary_b = _read_run_json(dir_b, "training_summary.json") or {}

    def headline(label: str, run_dir: Path, status: dict, summary: dict) -> None:
        best = summary.get("best_observed_dev_test_metrics") or {}
        score = best.get("selection_score")
        score_text = (
            f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        )
        print(
            f"{label}: {run_dir}\n"
            f"    state={status.get('state', '?')} "
            f"step={status.get('step', '?')} "
            f"best_selection={score_text}"
        )

    headline("A", dir_a, status_a, summary_a)
    headline("B", dir_b, status_b, summary_b)

    flat_a = _flatten_dict(config_a)
    flat_b = _flatten_dict(config_b)
    changed = sorted(
        key
        for key in set(flat_a) & set(flat_b)
        if flat_a[key] != flat_b[key]
    )
    only_a = sorted(set(flat_a) - set(flat_b))
    only_b = sorted(set(flat_b) - set(flat_a))

    print("")
    print(f"Config differences ({len(changed)} changed):")
    if not changed and not only_a and not only_b:
        print("  (configs are identical)")
    for key in changed:
        print(f"  {key}: {flat_a[key]!r} -> {flat_b[key]!r}")
    for key in only_a:
        print(f"  {key}: only in A ({flat_a[key]!r})")
    for key in only_b:
        print(f"  {key}: only in B ({flat_b[key]!r})")

    def metric_row(name: str, summary: dict) -> str:
        metrics = summary.get("best_observed_dev_test_metrics") or {}
        values = []
        for metric in (
            "global_f1_at_decision_threshold",
            "macro_game_f1_at_decision_threshold",
            "worst_game_f1_at_decision_threshold",
        ):
            value = metrics.get(
                metric,
                metrics.get(
                    metric.replace("_at_decision_threshold", "_tau099")
                ),
            )
            values.append(
                f"{metric}={value:.4f}"
                if isinstance(value, (int, float))
                else f"{metric}=n/a"
            )
        return f"  {name:<4s} " + " ".join(values)

    print("")
    print("Best observed dev-test metrics:")
    print(metric_row("A", summary_a))
    print(metric_row("B", summary_b))
    return 0


def cmd_run_export_tensorboard(args: argparse.Namespace) -> int:
    from game_cls.runs import read_training_metrics

    run_dir = _resolve_run_dir(args.target, Path(args.root))
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        print(
            "TensorBoard export requires the tensorboard package "
            f"(torch.utils.tensorboard unavailable: {exc}).",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out) if args.out else run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(out_dir))
    train_fields = (
        "loss",
        "ce",
        "threshold_loss",
        "interval_loss",
        "interval_ce",
        "interval_threshold_loss",
        "interval_threshold_weight",
        "interval_accuracy",
        "interval_positive_recall_tau099",
        "interval_negative_specificity_tau099",
        "interval_samples_per_second",
        "interval_step_time",
        "data_wait_ratio",
        "learning_rate",
        "grad_norm",
    )
    rows = read_training_metrics(run_dir)
    for row in rows:
        step = int(row.get("step", 0))
        for field in train_fields:
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                writer.add_scalar(f"train/{field}", float(value), step)
    # Unified evaluation history (train probe / validation / test).
    from game_cls.runs import read_evaluation_history

    evaluation_scalars = 0
    for record in read_evaluation_history(run_dir):
        step = int(record.get("step", 0))
        split = record.get("split", "validation")
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                writer.add_scalar(f"eval_{split}/{key}", float(value), step)
                evaluation_scalars += 1
    reports_dir = run_dir / "reports"
    evaluation_scalars = 0
    if reports_dir.is_dir():
        for metrics_file in sorted(reports_dir.glob("*/metrics.json")):
            payload = _read_run_json(metrics_file.parent, "metrics.json")
            if not payload:
                continue
            step = int(payload.get("checkpoint_step", 0))
            kind = payload.get("evaluation_kind", metrics_file.parent.name)
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    writer.add_scalar(f"eval_{kind}/{key}", float(value), step)
                    evaluation_scalars += 1
    writer.close()
    print(
        f"TensorBoard events written to {out_dir} "
        f"({len(rows)} train rows, {evaluation_scalars} eval scalars)."
    )
    print(f"View with: tensorboard --logdir {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# evaluate (standalone final test / validation evaluation)
# ---------------------------------------------------------------------------


_CHECKPOINT_ALIASES = (
    "last",
    "best_selection",
    "best_val_loss",
    "best_worst_game",
    "best_observed_dev_test_selection",
)


def _resolve_checkpoint_state(run_dir: Path, name: str):
    """Load a model state dict from a run's checkpoints directory.

    Evaluation only needs tensors, so all loads use ``weights_only=True``;
    internal training checkpoints are recognized by their marker and their
    model sub-dict is extracted.
    """
    import torch

    direct = Path(name)
    if direct.is_file():
        payload = torch.load(direct, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"], str(direct)
        return payload, str(direct)
    checkpoints = run_dir / "checkpoints"
    model_only = checkpoints / f"model_{name}.pth"
    if model_only.is_file():
        return (
            torch.load(model_only, map_location="cpu", weights_only=True),
            str(model_only),
        )
    full_state = checkpoints / f"checkpoint_{name}.pth"
    if full_state.is_file():
        payload = torch.load(
            full_state, map_location="cpu", weights_only=True
        )
        return payload["model"], str(full_state)
    available = sorted(
        path.name
        for path in checkpoints.glob("model_*.pth")
    ) if checkpoints.is_dir() else []
    raise SystemExit(
        f"Checkpoint '{name}' not found under {checkpoints}. "
        f"Available: {', '.join(available) or '(none)'}"
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate one checkpoint on the held-out test (or validation) split.

    The test split is evaluated exactly once, after training, and never
    participates in model selection: that is the whole point of the
    train/validation/test protocol.
    """
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError, split_role_warnings
    from game_cls.engine.checkpoint import unwrap_model
    from game_cls.engine.distributed import (
        cleanup_distributed,
        distributed_barrier,
        initialize_runtime,
        is_distributed,
    )
    from game_cls.engine.evaluator import evaluate
    from game_cls.engine.trainer import (
        _annotate_selection,
        _append_evaluation_history,
        _evaluation_history_record,
        build_eval_loader_for_split,
        has_independent_test,
    )
    from game_cls.model.builder import build_model
    from game_cls.reports.error_writer import (
        prepare_evaluation_directory,
        write_evaluation_report,
    )

    run_dir = _resolve_run_dir(args.run, Path(args.runs_root))
    config_source = args.config or str(run_dir / "resolved_config.json")
    if not Path(config_source).is_file():
        print(
            f"No config found for run {run_dir}: pass --config or ensure "
            f"{run_dir / 'resolved_config.json'} exists.",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_config(config_source)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    if config["data"].get("synthetic"):
        print(
            "evaluate requires real indexes; the run config uses "
            "synthetic data.",
            file=sys.stderr,
        )
        return 2
    if args.split == "test" and not has_independent_test(config):
        print(
            "This run has no independent test set: data.test_index was "
            "aliased as validation during training. Use --split "
            "validation, or retrain with a dedicated data.val_index.",
            file=sys.stderr,
        )
        return 3
    for warning in split_role_warnings(config):
        print(f"[WARNING] {warning}", file=sys.stderr)

    try:
        rank, world_size, local_rank, device = initialize_runtime(config)
    except RuntimeError as exc:
        print(f"Runtime initialization failed: {exc}", file=sys.stderr)
        return 1
    del local_rank
    try:
        state_dict, checkpoint_path = _resolve_checkpoint_state(
            run_dir, args.checkpoint
        )
        model = build_model(config["model"])
        unwrap_model(model).load_state_dict(state_dict, strict=True)
        model.to(device)
        if rank == 0:
            print(f"Loaded checkpoint: {checkpoint_path}")
        try:
            loader, components = build_eval_loader_for_split(
                config, args.split, rank, world_size
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        kind = "test_full" if args.split == "test" else "val_full"
        report_dir = (
            run_dir / "reports" / f"{kind}_{args.checkpoint}"
        )
        prepare_evaluation_directory(report_dir, rank)
        distributed_barrier()
        evaluation_cfg = config["evaluation"]
        loss_cfg = config["loss"]
        result = evaluate(
            unwrap_model(model),
            loader,
            device,
            config["decision"]["threshold"],
            checkpoint_step=0,
            distributed=is_distributed(),
            rank=rank,
            world_size=world_size,
            evaluation_kind=kind,
            report_dir=report_dir,
            full_auc_mode=evaluation_cfg.get("full_auc_mode", "histogram"),
            auc_histogram_bins=int(
                evaluation_cfg.get("auc_histogram_bins", 4096)
            ),
            amp=bool(
                evaluation_cfg.get(
                    "amp", config["device"].get("amp", False)
                )
            ),
            amp_dtype=str(
                evaluation_cfg.get(
                    "amp_dtype",
                    config["device"].get("amp_dtype", "bfloat16"),
                )
            ),
            parquet_row_group_size=int(
                evaluation_cfg.get("parquet_row_group_size", 4096)
            ),
            group_catalogs=getattr(
                getattr(loader, "dataset", None), "group_catalogs", None
            ),
            cross_entropy_weight=float(
                loss_cfg.get("cross_entropy_weight", 1.0)
            ),
            threshold_safety_margin=float(
                loss_cfg.get("threshold_safety_margin", 0.20)
            ),
            threshold_temperature=float(
                loss_cfg.get("threshold_temperature", 0.50)
            ),
            max_fpr_for_recall=float(
                evaluation_cfg.get("max_fpr_for_recall", 0.01)
            ),
            tail_calibration_enabled=bool(
                evaluation_cfg.get("tail_calibration_enabled", True)
            ),
        )
        distributed_barrier()
        if rank == 0:
            metrics = dict(result.metrics or {})
            metrics.update(
                {
                    "evaluation_kind": kind,
                    "evaluation_role": (
                        "test" if args.split == "test" else "validation"
                    ),
                    "evaluation_scope": "full",
                    "checkpoint": args.checkpoint,
                    "checkpoint_path": checkpoint_path,
                    "split": args.split,
                }
            )
            _annotate_selection(metrics, evaluation_cfg)
            write_evaluation_report(
                report_dir,
                metrics,
                result.grouped_metrics,
                merge_shards=True,
                lightweight=False,
                html_max_errors=int(
                    evaluation_cfg.get("html_max_errors_per_group", 200)
                ),
                preview_decoder=getattr(
                    getattr(loader, "dataset", None), "decoder", None
                ),
            )
            _append_evaluation_history(
                run_dir,
                _evaluation_history_record(
                    metrics,
                    kind=kind,
                    global_step=int(metrics.get("checkpoint_step", 0)),
                ),
            )
            summary_path = run_dir / (
                "test_evaluation.json"
                if args.split == "test"
                else "validation_evaluation.json"
            )
            summary_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("")
            print(f"=== {args.split} evaluation finished ===")
            print(f"checkpoint      : {checkpoint_path}")
            print(f"samples         : {metrics.get('sample_count')}")
            for key in (
                "selection_score",
                "global_f1_at_decision_threshold",
                "macro_game_f1_at_decision_threshold",
                "worst_game_f1_at_decision_threshold",
                "cross_entropy",
                "brier_score",
                "ece_20_bins",
            ):
                value = metrics.get(
                    key,
                    metrics.get(
                        key.replace("_at_decision_threshold", "_tau099")
                    ),
                )
                if isinstance(value, (int, float)):
                    print(f"{key:<22}: {value:.4f}")
            print(f"report          : {report_dir}")
        distributed_barrier()
        return 0
    finally:
        cleanup_distributed()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError

    failures = 0

    def check(ok: bool | None, label: str, detail: str = "") -> None:
        nonlocal failures
        mark = "[OK]" if ok else ("[WARN]" if ok is None else "[FAIL]")
        if ok is False:
            failures += 1
        suffix = f" — {detail}" if detail else ""
        print(f"{mark} {label}{suffix}")

    print(f"python: {sys.version.split()[0]} on {sys.platform}")
    try:
        import torch

        check(True, "torch importable", torch.__version__)
        check(
            torch.cuda.is_available() or None,
            "CUDA visible",
            "cuda.is_available()=" + str(torch.cuda.is_available()),
        )
    except ImportError as exc:
        check(False, "torch importable", str(exc))
    try:
        import torch_npu  # type: ignore

        check(True, "torch_npu importable", getattr(torch_npu, "__version__", "?"))
    except ImportError:
        check(None, "torch_npu importable", "not installed (needed only for NPU)")

    try:
        config = load_config(args.config, args.overrides)
        check(True, f"config schema valid: {args.config}")
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            check(False, "config schema", problem)
        return 1
    except FileNotFoundError as exc:
        check(False, "config file", str(exc))
        return 1

    accelerator = str(config["device"].get("accelerator", "auto"))
    if accelerator == "npu":
        from game_cls.runtime.npu_checks import probe_npu_environment

        for label, (ok, detail) in probe_npu_environment().items():
            check(ok, label, detail)
    else:
        check(
            None,
            "NPU device-operator probes",
            f"accelerator={accelerator}; run doctor with an npu config "
            "to probe bincount/scatter_add_/nonzero/index_select/GradScaler",
        )

    data_cfg = config["data"]
    if data_cfg.get("synthetic"):
        check(None, "data", "synthetic=true, index checks skipped")
    else:
        for key in ("train_index", "val_index", "test_index"):
            path = data_cfg.get(key)
            check(
                bool(path) and Path(path).is_file(),
                f"data.{key}",
                str(path),
            )
        for key in ("train_video_index", "val_video_index", "test_video_index"):
            path = data_cfg.get(key)
            check(
                bool(path) and Path(path).is_file(),
                f"data.{key}",
                str(path),
            )
        migration = data_cfg.get("split_migration") or {}
        if migration.get("test_used_as_validation"):
            check(
                None,
                "split roles",
                "test_index is aliased as validation; no independent "
                "test set (add data.val_index)",
            )
        else:
            check(True, "split roles", "train / validation / test")
        audit_path = data_cfg.get("audit_path")
        audit_exists = bool(audit_path) and Path(audit_path).is_file()
        audit_ok: bool | None = None
        if audit_exists:
            try:
                payload = json.loads(
                    Path(audit_path).read_text(encoding="utf-8")
                )
                audit_ok = bool(payload)
            except (json.JSONDecodeError, OSError):
                audit_ok = False
            check(
                audit_ok,
                "data.audit_path parses",
                str(audit_path),
            )
        else:
            check(
                False,
                "data.audit_path",
                str(audit_path) + " (run tools/audit_dataset.py)",
            )
        if data_cfg.get("backend") == "packed_uint8":
            for key in (
                "train_packed_index",
                "test_packed_index",
                "train_packed_video_index",
                "test_packed_video_index",
            ):
                path = data_cfg.get(key)
                check(
                    bool(path) and Path(path).is_file(),
                    f"data.{key}",
                    str(path),
                )
            val_packed_index = data_cfg.get("val_packed_index")
            if val_packed_index:
                check(
                    Path(val_packed_index).is_file(),
                    "data.val_packed_index",
                    str(val_packed_index),
                )

    model_cfg = config["model"]
    factory = str(model_cfg.get("factory", ""))
    if ":" in factory and "your_package" not in factory:
        module_name, _, function_name = factory.partition(":")
        try:
            import importlib

            module = importlib.import_module(module_name)
            check(
                callable(getattr(module, function_name, None)),
                "model.factory resolves",
                factory,
            )
        except ImportError as exc:
            check(False, "model.factory resolves", f"{factory}: {exc}")
    else:
        check(False, "model.factory", f"placeholder or malformed: {factory!r}")
    checkpoint_path = model_cfg.get("checkpoint_path")
    if checkpoint_path:
        check(
            Path(checkpoint_path).is_file(),
            "model.checkpoint_path exists",
            checkpoint_path,
        )
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.is_file():
            try:
                payload = torch.load(
                    checkpoint_file, map_location="cpu", weights_only=True
                )
                check(
                    isinstance(payload, dict) and len(payload) > 0,
                    "model.checkpoint parses (weights_only)",
                    f"{len(payload) if isinstance(payload, dict) else '?'} keys",
                )
            except Exception as exc:
                check(False, "model.checkpoint parses (weights_only)", str(exc))
    elif not data_cfg.get("synthetic"):
        check(False, "model.checkpoint_path", "not set (required for real data)")

    # Dummy forward: the model must build and return exactly [B, 2] logits.
    try:
        from game_cls.engine.device import autocast_context
        from game_cls.model.builder import build_model

        probe_model = build_model(dict(model_cfg))
        probe_model.eval()
        width = int(data_cfg.get("width", 448))
        height = int(data_cfg.get("height", 208))
        with torch.no_grad(), autocast_context(
            torch.device("cpu"), False, "bfloat16"
        ):
            # Same conversion the training loop applies: uint8 [B,2,3,H,W]
            # frames are scaled to float before the forward pass.
            dummy = torch.zeros(1, 2, 3, height, width, dtype=torch.uint8)
            dummy = dummy.to(torch.float32).div_(255.0)
            logits = probe_model(dummy[:, 0], dummy[:, 1])
        shape = tuple(logits.shape)
        check(
            shape == (1, 2),
            "model dummy forward returns [B, 2]",
            f"got {shape}",
        )
    except Exception as exc:
        check(False, "model dummy forward returns [B, 2]", str(exc)[:300])

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = bool(config.get("distributed", {}).get("enabled", False))
    try:
        from game_cls.runtime.distributed_runtime import (
            validate_launch_environment,
        )

        validate_launch_environment(config)
        check(
            True,
            "distributed",
            f"enabled={distributed} world_size={world_size}",
        )
    except Exception as exc:
        check(False, "distributed", str(exc))

    print("")
    print("FAIL" if failures else "PASS", f"({failures} failing checks)")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# dataset prepare / audit / pack
# ---------------------------------------------------------------------------


def _run_split_prepare(
    config: dict[str, Any],
    train_all_root: str | Path,
    test_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict:
    """Derive train/val from train_all_root and write the split bundle.

    Thin wrapper around ``indexing.write_split_bundle``; the first argument
    is the physical ``source_root`` (train_all) whose frames are logically
    re-partitioned into train/val.
    """
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import write_split_bundle

    data_config = config["data"]
    split = dict(data_config.get("split") or {})
    if output_dir is None:
        # Match tools/build_index.py's default index output dir; the split
        # manifest is then written at output_dir/manifest (e.g.
        # indexes/split_manifest.parquet).
        output_dir = Path("indexes")
    return write_split_bundle(
        train_all_root,
        test_root,
        output_dir,
        ImageSpec.from_config(data_config),
        ScanPolicy.from_config(data_config),
        DuplicatePolicy.from_config(data_config),
        split_config=split,
        compute_content_hash=True,
    )


def _maybe_prepare_split(config: dict[str, Any]) -> None:
    """Build the validation split on demand when ``prepare_if_missing``."""
    data_config = config["data"]
    if not data_config.get("prepare_if_missing"):
        return
    split = data_config.get("split") or {}
    if split.get("mode") != "from_train":
        return
    val_index = data_config.get("val_index")
    if val_index and Path(val_index).is_file():
        return  # the validation split already exists
    source_root = data_config.get("source_root")
    test_root = data_config.get("test_root")
    if not source_root or not test_root:
        raise SystemExit(
            "data.prepare_if_missing requires data.source_root and "
            "data.test_root to derive the validation split; set both, or "
            "run 'cls-trainer dataset prepare' separately first."
        )
    _run_split_prepare(config, source_root, test_root)


def cmd_dataset_prepare(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    data_config = config["data"]
    split = data_config.get("split") or {}
    if split.get("mode") != "from_train":
        print(
            "dataset prepare requires data.split.mode == 'from_train'; "
            f"got {split.get('mode')!r}. Configure data.split (mode, "
            "val_ratio, seed) to derive a validation split.",
            file=sys.stderr,
        )
        return 2
    if args.val_ratio is not None:
        split["val_ratio"] = args.val_ratio
    audit = _run_split_prepare(
        config,
        args.train_root,
        args.test_root,
        output_dir=args.output_dir,
    )
    summary = audit.get("split") or audit
    print("=== dataset prepare finished ===")
    print(f"split mode        : {split.get('mode')}")
    print(f"val ratio target  : {split.get('val_ratio')}")
    print(
        "val ratio achieved (delta=2): "
        f"{summary.get('val_ratio_achieved_delta2')}"
    )
    print(f"source videos     : {summary.get('source_video_count')}")
    print(f"manifest          : {split.get('manifest')}")
    return 0


def cmd_dataset_audit(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import audit_warning_messages, validate_audit

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    data_config = config["data"]
    source = Path(args.index_dir) / "audit.json"
    if not source.is_file():
        print(f"audit report not found: {source}", file=sys.stderr)
        return 2
    audit = json.loads(source.read_text(encoding="utf-8"))
    for split, report in audit["splits"].items():
        findings = report.get("findings", {})
        print(
            f"{split}: frames={report['frame_count']} videos={report['video_count']} "
            f"errors={len(findings.get('errors', []))} "
            f"warnings={len(findings.get('warnings', []))} "
            f"ignored={sum(findings.get('ignored', {}).get('counts', {}).values())}"
        )
    for warning in audit_warning_messages(audit):
        print(f"[WARNING] {warning}")
    if args.strict:
        validate_audit(
            audit,
            image_spec=ImageSpec.from_config(data_config),
            scan_policy=ScanPolicy.from_config(data_config),
            duplicate_policy=DuplicatePolicy.from_config(data_config),
            require_test_delta=int(config["pair"]["test_delta"]),
            require_content_hash=bool(
                data_config.get("require_content_hash_audit", False)
            ),
            require_unique_video_keys=bool(
                data_config.get(
                    "require_unique_video_keys_across_splits", False
                )
            ),
            minimum_pairs_per_game_label_delta={
                int(key): int(value)
                for key, value in data_config.get(
                    "minimum_pairs_per_game_label_delta", {}
                ).items()
            },
        )
        print("Strict dataset audit passed.")
    return 0


def cmd_dataset_pack(args: argparse.Namespace) -> int:
    from game_cls.config import load_config
    from game_cls.config_schema import ConfigSchemaError
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.packed_backend import pack_frame_index

    try:
        config = load_config(args.config, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2
    index_path = pack_frame_index(
        args.frame_index,
        args.output_dir,
        image_spec=ImageSpec.from_config(config["data"]),
        images_per_shard=args.images_per_shard,
    )
    print(
        json.dumps(
            {
                "packed_frame_index": str(index_path),
                "packed_video_index": str(
                    index_path.with_name("packed_video_entries.parquet")
                ),
                "manifest": str(
                    index_path.with_name("packed_manifest.json")
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cls-trainer",
        description="Dual-frame multi-game binary classification trainer.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser(
        "train", help="Train the classifier (unique run directory by default)."
    )
    train.add_argument(
        "--config",
        help="Config file. Optional with --resume (defaults to the run's "
        "resolved_config.json).",
    )
    train.add_argument(
        "--run-mode",
        choices=("unique", "fixed"),
        default="unique",
        help=(
            "unique (default): allocate an immutable timestamped run dir "
            "under experiment.output_dir. fixed: write into output_dir in "
            "place (legacy)."
        ),
    )
    train.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="Continue an existing run directory (implies fixed mode).",
    )
    train.add_argument(
        "--fork",
        metavar="RUN",
        help="Start a new run derived from an existing run's config "
        "(records parent lineage; implies unique mode).",
    )
    train.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Runs root used to resolve --fork targets by run id.",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan; initialize nothing.",
    )
    train.add_argument("overrides", nargs="*", metavar="key=value")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on the held-out test (or validation) "
        "split; the test split never participates in model selection.",
    )
    evaluate.add_argument(
        "--run",
        required=True,
        help="Run directory, run id or 'latest'.",
    )
    evaluate.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Runs root used to resolve --run by run id.",
    )
    evaluate.add_argument(
        "--checkpoint",
        default="best_selection",
        help=(
            "Checkpoint alias (last, best_selection, best_val_loss, "
            "best_worst_game) or a direct .pth path."
        ),
    )
    evaluate.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Split to evaluate; 'test' requires an independent test set.",
    )
    evaluate.add_argument(
        "--config",
        help="Optional config override (defaults to the run's "
        "resolved_config.json).",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    config = subparsers.add_parser("config", help="Config utilities.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    show = config_sub.add_parser(
        "show", help="Print the resolved (finalized) configuration."
    )
    show.add_argument("--config", required=True)
    show.add_argument(
        "--with-source",
        action="store_true",
        help="Annotate every value with the file/override that set it.",
    )
    show.add_argument("overrides", nargs="*", metavar="key=value")
    show.set_defaults(func=cmd_config_show)
    validate = config_sub.add_parser(
        "validate", help="Load and validate a config without training."
    )
    validate.add_argument("--config", required=True)
    validate.add_argument("overrides", nargs="*", metavar="key=value")
    validate.set_defaults(func=cmd_config_validate)
    reference = config_sub.add_parser(
        "reference", help="List every known config key with its meaning."
    )
    reference.set_defaults(func=cmd_config_reference)

    run = subparsers.add_parser("run", help="Inspect recorded runs.")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_list = run_sub.add_parser("list", help="List runs from the index.")
    run_list.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_list.add_argument("--limit", type=int, default=20)
    run_list.set_defaults(func=cmd_run_list)
    run_show = run_sub.add_parser(
        "show", help="Show one run ('latest' or a run directory)."
    )
    run_show.add_argument("target", help="'latest', a run directory or run id.")
    run_show.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_show.set_defaults(func=cmd_run_show)
    run_compare = run_sub.add_parser(
        "compare", help="Compare two runs: config diff + best metrics."
    )
    run_compare.add_argument("run_a", help="Run directory, run id or 'latest'.")
    run_compare.add_argument("run_b", help="Run directory, run id or 'latest'.")
    run_compare.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_compare.set_defaults(func=cmd_run_compare)
    run_export = run_sub.add_parser(
        "export-tensorboard",
        help="Export a run's curves to TensorBoard event files.",
    )
    run_export.add_argument("target", help="Run directory, run id or 'latest'.")
    run_export.add_argument("--out", help="Output logdir (default <run>/tensorboard).")
    run_export.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_export.set_defaults(func=cmd_run_export_tensorboard)

    doctor = subparsers.add_parser(
        "doctor", help="Check environment, config, data and model readiness."
    )
    doctor.add_argument("--config", required=True)
    doctor.add_argument("overrides", nargs="*", metavar="key=value")
    doctor.set_defaults(func=cmd_doctor)

    init = subparsers.add_parser(
        "init", help="Create a minimal recipe for a given profile."
    )
    init.add_argument(
        "--profile",
        default="npu_8p",
        help="Machine profile (see configs/profiles/).",
    )
    init.add_argument("--name", default="my_run", help="Run name.")
    init.add_argument(
        "--output",
        default="configs/recipes/my_run.yaml",
        help="Recipe file to create.",
    )
    init.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    init.set_defaults(func=cmd_init)

    dataset = subparsers.add_parser(
        "dataset", help="Dataset preparation, audit and packing."
    )
    dataset_sub = dataset.add_subparsers(
        dest="dataset_command", required=True
    )
    prepare = dataset_sub.add_parser(
        "prepare",
        help="Derive a source-video-level validation split from train_root.",
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--train-root", required=True)
    prepare.add_argument("--test-root", required=True)
    prepare.add_argument("--output-dir", default="indexes")
    prepare.add_argument(
        "--val-ratio",
        type=float,
        help="Override data.split.val_ratio for this preparation.",
    )
    prepare.add_argument("overrides", nargs="*", metavar="key=value")
    prepare.set_defaults(func=cmd_dataset_prepare)
    audit = dataset_sub.add_parser(
        "audit", help="Validate an index audit report."
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--index-dir", default="indexes")
    audit.add_argument(
        "--strict",
        action="store_true",
        help="Run the strict validate_audit gate (mirrors "
        "tools/audit_dataset.py).",
    )
    audit.add_argument("overrides", nargs="*", metavar="key=value")
    audit.set_defaults(func=cmd_dataset_audit)
    pack = dataset_sub.add_parser(
        "pack", help="Pack decoded CHW uint8 frames into memmapped shards."
    )
    pack.add_argument("--config", required=True)
    pack.add_argument("--frame-index", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--images-per-shard", type=int, default=4096)
    pack.add_argument("overrides", nargs="*", metavar="key=value")
    pack.set_defaults(func=cmd_dataset_pack)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    known = {
        "train",
        "config",
        "run",
        "doctor",
        "init",
        "evaluate",
        "dataset",
    }
    if not argv or argv[0] not in known:
        # cls-trainer --config x.yaml k=v  =>  cls-trainer train --config ...
        argv = ["train", *argv]
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def train_command_main() -> int:
    """Entry point for tools/train.py (train subcommand only)."""
    argv = ["train", *sys.argv[1:]]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
