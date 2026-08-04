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

    def __enter__(self) -> "_TeeContext":
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.stdout, self.stderr
        return self

    def __exit__(self, *exc_info: Any) -> None:
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
        self.stdout.close()
        self.stderr.close()


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
    quick_every = int(evaluation_cfg.get("quick_test_every_steps", 0))
    full_every = int(evaluation_cfg.get("full_test_every_steps", 0))

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
        f"quick test         : "
        f"{'every ' + str(quick_every) + ' steps' if quick_every else 'disabled'}"
    )
    print(
        f"full test          : "
        f"{'every ' + str(full_every) + ' steps' if full_every else 'disabled'}"
        f" (at end: {bool(evaluation_cfg.get('full_test_at_end', True))})"
    )
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
        else:
            if not args.config:
                raise SystemExit("train requires --config (or --resume).")
            config_source = args.config
        config = load_config(config_source, args.overrides)
    except ConfigSchemaError as exc:
        for problem in exc.problems:
            print(f"Config error: {problem}", file=sys.stderr)
        return 2

    if args.resume:
        config["experiment"]["output_dir"] = str(resume_dir)
        config["experiment"]["run_mode"] = "fixed"
        config["train"]["resume_path"] = _resolve_resume_checkpoint(
            resume_dir, config
        )
    else:
        config["experiment"]["run_mode"] = args.run_mode

    if args.dry_run:
        return _dry_run_report(config, config_source)

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
    from game_cls.runs import read_run_index

    records: list[dict[str, Any]] = []
    records.extend(read_run_index(root))
    for nested in sorted(root.glob("*/index.jsonl")):
        records.extend(read_run_index(nested.parent))
    return records


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
    target = args.target
    run_dir: Path | None = None
    if target != "latest":
        candidate = Path(target)
        if not candidate.is_dir():
            raise SystemExit(f"Run directory not found: {target}")
        run_dir = candidate
    else:
        records = _find_index_records(root)
        if records:
            run_dir = Path(records[-1]["output_dir"])
        else:
            candidates = sorted(
                path for path in root.glob("*/*") if (path / "status.json").is_file()
            )
            if not candidates:
                print(f"No runs found under {root.resolve()}.")
                return 1
            run_dir = candidates[-1]
    assert run_dir is not None

    def read_json(name: str) -> dict[str, Any] | None:
        path = run_dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    print(f"run directory: {run_dir}")
    manifest = read_json("manifest.json")
    status = read_json("status.json")
    summary = read_json("training_summary.json")
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
    if status:
        print(f"state        : {status.get('state')}")
        print(f"last update  : {status.get('last_update')}")
        if status.get("error_type"):
            print(f"error        : {status['error_type']}: {status.get('error_message')}")
    if summary:
        best = summary.get("best_observed_dev_test_metrics") or {}
        if isinstance(best.get("selection_score"), (int, float)):
            print(f"best score   : {best['selection_score']:.4f}")
    for artifact in ("summary.md", "train_metrics.jsonl", "console.log"):
        if (run_dir / artifact).is_file():
            print(f"artifact     : {run_dir / artifact}")
    return 0


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

    data_cfg = config["data"]
    if data_cfg.get("synthetic"):
        check(None, "data", "synthetic=true, index checks skipped")
    else:
        for key in ("train_index", "test_index"):
            path = data_cfg.get(key)
            check(
                bool(path) and Path(path).is_file(),
                f"data.{key}",
                str(path),
            )
        audit_path = data_cfg.get("audit_path")
        check(
            bool(audit_path) and Path(audit_path).is_file() or None,
            "data.audit_path",
            str(audit_path) + (" (run tools/audit_dataset.py)" if not (audit_path and Path(audit_path).is_file()) else ""),
        )
        if data_cfg.get("backend") == "packed_uint8":
            for key in ("train_packed_index", "test_packed_index"):
                path = data_cfg.get(key)
                check(
                    bool(path) and Path(path).is_file(),
                    f"data.{key}",
                    str(path),
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
        check(True, "model.checkpoint_path exists", checkpoint_path)
    elif not data_cfg.get("synthetic"):
        check(False, "model.checkpoint_path", "not set (required for real data)")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = bool(config.get("distributed", {}).get("enabled", False))
    if distributed and world_size <= 1:
        check(
            False,
            "distributed",
            f"distributed.enabled=true but WORLD_SIZE={world_size}; launch via torchrun",
        )
    else:
        check(True, "distributed", f"enabled={distributed} world_size={world_size}")

    print("")
    print("FAIL" if failures else "PASS", f"({failures} failing checks)")
    return 1 if failures else 0


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
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan; initialize nothing.",
    )
    train.add_argument("overrides", nargs="*", metavar="key=value")
    train.set_defaults(func=cmd_train)

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
    run_show.add_argument("target", help="'latest' or a run directory path.")
    run_show.add_argument("--root", default=DEFAULT_RUNS_ROOT)
    run_show.set_defaults(func=cmd_run_show)

    doctor = subparsers.add_parser(
        "doctor", help="Check environment, config, data and model readiness."
    )
    doctor.add_argument("--config", required=True)
    doctor.add_argument("overrides", nargs="*", metavar="key=value")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    known = {"train", "config", "run", "doctor"}
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
