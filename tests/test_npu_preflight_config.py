"""The NPU preflight config must stay data-free (step6 P0-4).

The npu_1p / npu_8p configs point at production indexes and checkpoints, so
they can only run on a provisioned runner -- a fresh Ascend host could not
tell "the NPU stack is broken" from "this box has no data". The preflight
config exists to be runnable anywhere, and the workflow's repeatable gate
depends on that. These checks fail if it drifts back toward production data.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from game_cls.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = "configs/npu_synthetic_smoke.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "npu-ci.yml"


class NpuPreflightConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PREFLIGHT)

    def test_targets_the_npu_accelerator_with_bf16(self) -> None:
        # A preflight that silently ran on CPU would prove nothing.
        self.assertEqual(self.config["device"]["accelerator"], "npu")
        self.assertTrue(self.config["device"]["amp"])
        self.assertEqual(self.config["device"]["amp_dtype"], "bfloat16")

    def test_needs_no_dataset_on_disk(self) -> None:
        data = self.config["data"]
        self.assertTrue(data["synthetic"])
        self.assertFalse(data["prepare_if_missing"])
        # Audit/provenance contracts have nothing to check against synthetic
        # data, so leaving them on would fail before the device is touched.
        self.assertFalse(data["strict_audit"])
        self.assertFalse(data["require_content_hash_audit"])
        self.assertFalse(data["require_independent_test"])

    def test_needs_no_checkpoint_or_sidecar(self) -> None:
        self.assertIsNone(self.config["model"]["checkpoint_path"])
        # Subtype grouping needs a metadata sidecar; asking for it here would
        # make constrained selection reject every checkpoint.
        self.assertFalse(self.config["evaluation"]["group_by_negative_subtype"])
        self.assertIsNone(self.config["evaluation"]["max_worst_subtype_fpr"])
        self.assertFalse((self.config["data"].get("hard_negative") or {})["enabled"])

    def test_stays_a_smoke_test(self) -> None:
        # The preflight runs on every schedule tick; it must stay short.
        self.assertLessEqual(int(self.config["train"]["max_steps"]), 100)


class NpuWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        yaml = self._yaml()
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        # PyYAML parses the unquoted `on:` key as the boolean True.
        self.triggers = self.workflow.get("on") or self.workflow[True]
        self.jobs = self.workflow["jobs"]

    @staticmethod
    def _yaml():
        try:
            import yaml
        except ImportError:  # pragma: no cover - environment guard
            raise unittest.SkipTest("PyYAML is not installed") from None
        return yaml

    def test_preflight_runs_unattended_but_production_does_not(self) -> None:
        self.assertIn("schedule", self.triggers)
        self.assertIn("workflow_dispatch", self.triggers)
        for name in ("device-ops", "preflight-1p", "preflight-8p"):
            self.assertNotIn("if", self.jobs[name], f"{name} must run on schedule")
        for name in ("npu-1p", "npu-8p"):
            condition = self.jobs[name]["if"]
            self.assertIn("workflow_dispatch", condition)
            self.assertIn("run_production", condition)

    def test_production_jobs_check_prerequisites_before_training(self) -> None:
        for name in ("npu-1p", "npu-8p"):
            steps = self.jobs[name]["steps"]
            runs = [str(step.get("run", "")) for step in steps]
            doctor = next(i for i, run in enumerate(runs) if "doctor" in run)
            trains = [i for i, run in enumerate(runs) if "tools/train.py" in run]
            self.assertTrue(trains, f"{name} runs no training")
            self.assertLess(doctor, min(trains), f"{name} trains before doctor")

    def test_preflight_uses_the_synthetic_config_only(self) -> None:
        for name in ("preflight-1p", "preflight-8p"):
            body = "\n".join(
                str(step.get("run", "")) for step in self.jobs[name]["steps"]
            )
            self.assertIn("npu_synthetic_smoke.yaml", body)
            self.assertNotIn("npu_1p.yaml", body)
            self.assertNotIn("npu_8p.yaml", body)

    def test_pip_install_never_replaces_torch_npu(self) -> None:
        # A plain `pip install -e .` re-resolves torch and can clobber the
        # runner's CANN-matched build, which then fails to see the NPU.
        for name, job in self.jobs.items():
            for step in job["steps"]:
                run = str(step.get("run", ""))
                if "pip install" in run:
                    self.assertIn("--no-deps", run, f"{name}: {run!r}")

    def test_devices_are_not_shared_between_concurrent_runs(self) -> None:
        self.assertEqual(self.workflow["concurrency"]["group"], "npu-acceptance")
        self.assertFalse(self.workflow["concurrency"]["cancel-in-progress"])


if __name__ == "__main__":
    unittest.main()
