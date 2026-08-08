"""cls-trainer command line interface.

Workflows:

    cls-trainer train --config configs/recipes/game_cls_production.yaml [key=value ...]
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
        # Audit P0-9: existence of every index file does NOT prove they are one
        # generation -- a bundle whose train came from a new prepare and whose
        # test was left from an older one passes every check above. The bundle
        # manifest is the commit record that settles it.
        #
        # Scoped to split.mode == "from_train": only that path publishes a
        # commit record. An index directory built by write_index_bundle is a
        # different, legitimate shape with no split summary or split manifest,
        # so it is reported as not-applicable rather than failed.
        if (data_cfg.get("split") or {}).get("mode") == "from_train":
            from game_cls.data.indexing import SplitBundleError, verify_split_bundle

            bundle_dir = Path(
                data_cfg.get("train_index") or "indexes/train_frames.parquet"
            ).parent
            try:
                bundle = verify_split_bundle(bundle_dir)
                check(
                    True,
                    "index bundle generation",
                    f"bundle_id={(bundle or {}).get('bundle_id', '?')}",
                )
            except SplitBundleError as exc:
                # First line only: the full guidance is long, and doctor prints
                # one line per check.
                check(False, "index bundle generation", str(exc).splitlines()[0])
        else:
            check(
                None,
                "index bundle generation",
                "data.split.mode is not 'from_train'; no split bundle to verify",
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
        identity_mode = (data_cfg.get("source_video_identity") or {}).get(
            "mode", "game_video"
        )
        check(
            identity_mode in ("game_video", "game_label_video"),
            "data.source_video_identity.mode",
            str(identity_mode),
        )
        if identity_mode == "game_label_video":
            check(
                None,
                "source identity warning",
                "identical video_id values under different labels are "
                "assumed to be physically unrelated source videos; if "
                "false, train/validation leakage may occur",
            )
        split_cfg = data_cfg.get("split") or {}
        manifest_name = split_cfg.get("manifest")
        if manifest_name is None:
            check(
                None,
                "split manifest identity mode",
                "data.split.manifest not configured",
            )
        else:
            index_dir = (
                Path(data_cfg["audit_path"]).parent
                if data_cfg.get("audit_path")
                else Path("indexes")
            )
            manifest_path = Path(manifest_name)
            if not manifest_path.is_absolute():
                manifest_path = index_dir / manifest_path
            if not manifest_path.is_file():
                check(
                    None,
                    "split manifest identity mode",
                    f"no manifest to cross-check ({manifest_path}); run "
                    "dataset prepare first",
                )
            else:
                try:
                    from game_cls.data.splitter import load_split_manifest

                    stored = load_split_manifest(manifest_path).get(
                        "split_source_identity_mode"
                    )
                    check(
                        stored == identity_mode,
                        "split manifest identity mode",
                        f"manifest={stored} config={identity_mode}",
                    )
                except Exception as exc:  # noqa: BLE001 - doctor reports any load failure
                    check(False, "split manifest identity mode", str(exc))
        audit_path = data_cfg.get("audit_path")
        audit_exists = bool(audit_path) and Path(audit_path).is_file()
        audit_ok: bool | None = None
        if audit_exists:
            try:
                payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
                audit_ok = bool(payload)
            except (json.JSONDecodeError, OSError):
                audit_ok = False
            check(
                audit_ok,
                "data.audit_path parses",
                str(audit_path),
            )
            if audit_ok:
                from game_cls.config_schema import resolve_source_identity_namespaces

                stored_namespaces = (payload.get("leakage") or {}).get(
                    "source_identity_namespaces"
                )
                configured = resolve_source_identity_namespaces(
                    data_cfg.get("source_video_identity")
                )
                check(
                    (stored_namespaces or {}) == configured,
                    "audit source identity namespaces",
                    f"audit={stored_namespaces or {}} config={configured}",
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
        with torch.no_grad(), autocast_context(torch.device("cpu"), False, "bfloat16"):
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
