"""Model export + release gate (step5 P6)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ExportVerifyTests(unittest.TestCase):
    def _authorize_export(self, run_dir: Path, config_path: Path | None = None) -> None:
        from game_cls.release import release_identity
        from game_cls.reports.benchmark import (
            check_gates,
            file_sha256,
            write_benchmark_report,
        )

        path = config_path or (run_dir / "resolved_config.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        challenge = path.parent / "challenge.identity"
        challenge.write_bytes(b"fixed challenge bundle")
        config["data"]["challenge_index"] = str(challenge)
        gates = {"global_fpr_at_decision_threshold": {"op": "<=", "value": 0.01}}
        benchmark_dir = path.parent / "benchmarks"
        config["benchmark"]["gate_metrics"] = gates
        config["benchmark"]["output_dir"] = str(benchmark_dir)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        checkpoint = run_dir / "checkpoints" / "model_last.pth"
        if not checkpoint.is_file():
            checkpoint = run_dir / "checkpoints" / "checkpoint_last.pth"
        identity, components = release_identity(
            run_id=run_dir.name,
            checkpoint_sha256=file_sha256(checkpoint),
            config=config,
        )
        metrics = {"global_fpr_at_decision_threshold": 0.0}
        write_benchmark_report(
            benchmark_dir,
            run_id=run_dir.name,
            checkpoint_alias="last",
            metrics=metrics,
            gates=check_gates(metrics, gates),
            grouped_metrics=None,
            gate_metrics=gates,
            checkpoint_sha256=identity["checkpoint_sha256"],
            resolved_config_sha256=identity["resolved_config_sha256"],
            challenge_dataset_fingerprint=identity["challenge_dataset_fingerprint"],
            challenge_bundle_components=components,
            gate_spec_fingerprint=identity["gate_spec_fingerprint"],
        )

    def _trained_run(self, directory: str) -> Path:
        from game_cls.config import load_config
        from game_cls.engine.trainer import run_training

        config = load_config("configs/recipes/example_debug.yaml")
        config["experiment"]["output_dir"] = str(Path(directory) / "run")
        config["train"].update(
            {
                "max_steps": 2,
                "steps_per_epoch": 16,
                "local_batch_size": 4,
                "log_every_steps": 8,
            }
        )
        config["evaluation"].update(
            {
                "train_probe_every_steps": 0,
                "val_quick_every_steps": 0,
                "val_full_every_steps": 0,
                "val_full_at_end": False,
            }
        )
        config["checkpoint"].update({"save_topk": 0})
        config.pop("early_stopping", None)
        result = run_training(config)
        return Path(result["output_dir"])

    def _artifact_dir(self, export_dir: Path, run_dir: Path) -> Path:
        """The immutable per-artifact export directory (audit PR-F):
        ``export_dir/<run_id>/<checkpoint_sha256>/``."""
        run_export = export_dir / run_dir.name
        self.assertTrue(run_export.is_dir(), f"no run dir under {export_dir}")
        (sha_dir,) = list(run_export.iterdir())
        return sha_dir

    def test_weights_export_roundtrip(self) -> None:
        import torch

        from game_cls.cli.export import cmd_export
        from game_cls.model.builder import build_model

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._trained_run(directory)
            export_dir = Path(directory) / "exports"
            args = type(
                "Args",
                (),
                {
                    "run": str(run_dir),
                    "runs_root": str(Path(directory) / "runs"),
                    "config": None,
                    "checkpoint": "last",
                    "skip_gate": True,
                    "format": "weights",
                    "out": str(export_dir),
                },
            )()
            self._authorize_export(run_dir)
            rc = cmd_export(args)
            self.assertEqual(rc, 0)
            artifact_dir = self._artifact_dir(export_dir, run_dir)
            weights = artifact_dir / "model.pt"
            manifest = json.loads(
                (artifact_dir / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(weights.is_file())
            self.assertEqual(manifest["decision.threshold"], 0.99)
            self.assertEqual(manifest["shape"], [1, 2, 3, 208, 448])
            # The exported state dict round-trips into a fresh model.
            model = build_model(
                {"factory": "game_cls.model.builder:build_demo_model", "num_classes": 2}
            )
            model.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True),
                strict=True,
            )

    def test_export_refuses_overwriting_an_existing_artifact(self) -> None:
        # Audit PR-F: the export layout is immutable. Re-exporting the same
        # checkpoint resolves to the same <run_id>/<sha> directory and must be
        # refused instead of silently overwriting the deployment artifact.
        from game_cls.cli.export import cmd_export

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._trained_run(directory)
            export_dir = Path(directory) / "exports"
            args = type(
                "Args",
                (),
                {
                    "run": str(run_dir),
                    "runs_root": str(Path(directory) / "runs"),
                    "config": None,
                    "checkpoint": "last",
                    "skip_gate": True,
                    "format": "weights",
                    "out": str(export_dir),
                },
            )()
            self._authorize_export(run_dir)
            self.assertEqual(cmd_export(args), 0)
            self.assertEqual(cmd_export(args), 2)

    def _export_config(self, base: Path, **data_overrides) -> Path:
        """Write a resolved config snapshot with data.* overridden."""
        from game_cls.config import load_config

        config = load_config("configs/recipes/example_debug.yaml")
        config["data"].update(data_overrides)
        path = base / "export_config.json"
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def test_manifest_shape_follows_the_configured_frame_geometry(self) -> None:
        # The shape used to be hardcoded to [1, 2, 3, 208, 448], so an
        # overridden task profile silently produced a wrong manifest. None of
        # these three numbers matches the old literal.
        from game_cls.cli.export import cmd_export

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_dir = self._trained_run(directory)
            export_dir = base / "exports"
            config = self._export_config(base, width=96, height=48, channels=1)
            args = type(
                "Args",
                (),
                {
                    "run": str(run_dir),
                    "runs_root": str(base / "runs"),
                    "config": str(config),
                    "checkpoint": "last",
                    "skip_gate": True,
                    "format": "weights",
                    "out": str(export_dir),
                },
            )()
            self._authorize_export(run_dir, config)
            self.assertEqual(cmd_export(args), 0)
            artifact_dir = self._artifact_dir(export_dir, run_dir)
            manifest = json.loads(
                (artifact_dir / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["shape"], [1, 2, 1, 48, 96])

    def _export_via_cli(self, run_dir: Path, base: Path, config: Path) -> int:
        """Run ``export`` with neither --out nor --format, as a user would."""
        from game_cls.cli import main

        self._authorize_export(run_dir, config)

        return main(
            [
                "export",
                "--run",
                str(run_dir),
                "--runs-root",
                str(base / "runs"),
                "--config",
                str(config),
                "--checkpoint",
                "last",
            ]
        )

    def test_export_out_dir_comes_from_the_config(self) -> None:
        # argparse used to default --out to "exports", which shadowed
        # export.output_dir on every plain call.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_dir = self._trained_run(directory)
            configured_dir = base / "from_config"
            config = self._export_config(base)
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["export"]["output_dir"] = str(configured_dir)
            config.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.assertEqual(self._export_via_cli(run_dir, base, config), 0)
            self.assertTrue(
                (self._artifact_dir(configured_dir, run_dir) / "export_manifest.json").is_file()
            )
            # The CWD-relative default "exports" was never touched.
            self.assertFalse((Path("exports") / run_dir.name).exists())

    def test_export_format_comes_from_the_config(self) -> None:
        # "onnx" is the only non-default value the schema allows, so it is the
        # only way to prove the config wins: with the old --format default of
        # "weights" this run would silently take the weights branch.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_dir = self._trained_run(directory)
            out_dir = base / "from_config"
            config = self._export_config(base)
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["export"]["output_dir"] = str(out_dir)
            payload["export"]["format"] = "onnx"
            config.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            code = self._export_via_cli(run_dir, base, config)
            # Asserted via the artifacts rather than the exit code, because
            # whether onnxruntime is installed decides between a traced graph
            # (0) and the friendly "install it or use --format weights" (2).
            # Either way the weights branch must not have run.
            if code == 0:
                artifact_dir = self._artifact_dir(out_dir, run_dir)
                self.assertFalse((artifact_dir / "model.pt").is_file())
                self.assertTrue((artifact_dir / "model.onnx").is_file())
            else:
                self.assertEqual(code, 2)
                # Transactional export publishes nothing on a failed trace.
                self.assertEqual(list((out_dir / run_dir.name).iterdir()), [])

    def test_manifest_records_base_checkpoint_sha_and_metric_summary(self) -> None:
        from game_cls.cli.export import cmd_export
        from game_cls.runs import sha256_file

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_dir = self._trained_run(directory)
            # A pretrained backbone the run started from, hashed on demand
            # because this run's manifest has no base_checkpoint_sha256.
            pretrained = base / "backbone.pth"
            pretrained.write_bytes(b"pretend-pretrained-weights")
            config = self._export_config(base)
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["model"]["checkpoint_path"] = str(pretrained)
            payload["evaluation"]["selection_metric"] = (
                "worst_game_f1_at_decision_threshold"
            )
            config.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Known metrics in the run's summary so the manifest can be
            # checked against exact values rather than mere presence.
            (run_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "global_step": 8,
                        "best_validation_metrics": {
                            "worst_game_f1_at_decision_threshold": 0.8125,
                            "global_fpr_at_decision_threshold": 0.0025,
                            "confidence_histogram": {"0.5": 3},
                        },
                        "last_checkpoint_metrics": {
                            "global_fpr_at_decision_threshold": 0.0075
                        },
                    }
                ),
                encoding="utf-8",
            )
            export_dir = base / "exports"
            args = type(
                "Args",
                (),
                {
                    "run": str(run_dir),
                    "runs_root": str(base / "runs"),
                    "config": str(config),
                    "checkpoint": "last",
                    "skip_gate": True,
                    "format": "weights",
                    "out": str(export_dir),
                },
            )()
            self._authorize_export(run_dir, config)
            self.assertEqual(cmd_export(args), 0)
            artifact_dir = self._artifact_dir(export_dir, run_dir)
            manifest = json.loads(
                (artifact_dir / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["base_checkpoint_sha256"], sha256_file(pretrained)
            )
            summary = manifest["metric_summary"]
            self.assertEqual(
                summary["selection_metric"], "worst_game_f1_at_decision_threshold"
            )
            self.assertEqual(summary["selection_metric_value"], 0.8125)
            self.assertEqual(summary["global_step"], 8)
            self.assertEqual(
                summary["best_validation"]["global_fpr_at_decision_threshold"], 0.0025
            )
            self.assertEqual(
                summary["last_checkpoint"]["global_fpr_at_decision_threshold"], 0.0075
            )
            # Nested tables stay in training_summary.json.
            self.assertNotIn("confidence_histogram", summary["best_validation"])

    def test_manifest_prefers_the_recorded_base_checkpoint_sha(self) -> None:
        # The run already hashed its base checkpoint; that value is
        # authoritative even if the file later moved or changed on disk.
        from game_cls.cli.export import cmd_export
        from game_cls.runs import read_manifest, write_manifest

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_dir = self._trained_run(directory)
            run_manifest = read_manifest(run_dir) or {}
            run_manifest["base_checkpoint_sha256"] = "f" * 64
            write_manifest(run_dir, run_manifest)
            moved = base / "gone.pth"
            moved.write_bytes(b"different-bytes")
            config = self._export_config(base)
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["model"]["checkpoint_path"] = str(moved)
            config.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            export_dir = base / "exports"
            args = type(
                "Args",
                (),
                {
                    "run": str(run_dir),
                    "runs_root": str(base / "runs"),
                    "config": str(config),
                    "checkpoint": "last",
                    "skip_gate": True,
                    "format": "weights",
                    "out": str(export_dir),
                },
            )()
            self._authorize_export(run_dir, config)
            self.assertEqual(cmd_export(args), 0)
            artifact_dir = self._artifact_dir(export_dir, run_dir)
            manifest = json.loads(
                (artifact_dir / "export_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["base_checkpoint_sha256"], "f" * 64)

    def test_onnx_manifest_carries_the_run_id(self) -> None:
        # The ONNX path wrote run_id="" while the weights path wrote
        # run_dir.name; both now go through _export_manifest.
        from game_cls.cli.export import _export_manifest

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "20260806-120000-demo"
            run_dir.mkdir(parents=True)
            manifest = _export_manifest(
                run_dir=run_dir,
                config={
                    "data": {"width": 96, "height": 48, "channels": 1},
                    "model": {"factory": "pkg.mod:factory", "checkpoint_path": None},
                    "evaluation": {},
                },
                alias="best_selection",
                checkpoint_path=str(run_dir / "checkpoints" / "model_x.pth"),
                manifest_extra={"decision.threshold": 0.99},
                input_shape=(1, 2, 1, 48, 96),
                artifact={"onnx": "model.onnx", "verification_max_abs_diff": 1e-7},
            )
            self.assertEqual(manifest["run_id"], "20260806-120000-demo")
            self.assertEqual(manifest["shape"], [1, 2, 1, 48, 96])
            self.assertIn("base_checkpoint_sha256", manifest)
            self.assertFalse(manifest["metric_summary"]["available"])


class ReleaseGateTests(unittest.TestCase):
    def test_release_gate_fails_on_template(self) -> None:
        from game_cls.cli.config_tools import cmd_config_validate

        args = type(
            "Args",
            (),
            {
                "config": "configs/recipes/game_cls_production.yaml",
                "overrides": [],
                "release": True,
            },
        )()
        self.assertEqual(cmd_config_validate(args), 2)

    def test_release_gate_passes_with_baseline(self) -> None:
        import tempfile

        import yaml

        from game_cls.cli.config_tools import cmd_config_validate
        from game_cls.reports.benchmark import check_gates, write_benchmark_report

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = yaml.safe_load(
                Path("configs/recipes/game_cls_production.yaml").read_text(
                    encoding="utf-8"
                )
            )
            config.setdefault("evaluation", {})["minimum_worst_game_f1"] = 0.75
            gate_metrics = config.setdefault("benchmark", {})["gate_metrics"] = {
                "global_fpr_at_decision_threshold": 0.01
            }
            # Audit P0-3: a release-ready check with NO benchmark must fail, so
            # this passing case needs an actual report.
            config["benchmark"]["output_dir"] = str(base / "benchmarks")
            metrics = {
                "sample_count": 100,
                "global_fpr_at_decision_threshold": 0.004,
            }
            write_benchmark_report(
                base / "benchmarks",
                run_id="run-1",
                checkpoint_alias="best_selection",
                metrics=metrics,
                gates=check_gates(metrics, gate_metrics),
                grouped_metrics=None,
                gate_metrics=gate_metrics,
            )
            cfg_path = base / "release.yaml"
            cfg_path.write_text(
                yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
            )
            args = type(
                "Args",
                (),
                {"config": str(cfg_path), "overrides": [], "release": True},
            )()
            self.assertEqual(cmd_config_validate(args), 0)


if __name__ == "__main__":
    unittest.main()
