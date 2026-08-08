"""Release identity gate: a PASS binds to the FULL identity, not the weights.

Audit P0-2. ``release check`` used to accept any benchmark report whose
``checkpoint_sha256`` matched the artifact being released. But FPR, recall and
selection eligibility are functions of model + threshold + challenge set +
preprocessing -- not of the weights alone. Two runs can share one checkpoint
byte-for-byte and still have genuinely different benchmark results, so matching
the hash alone let a PASS earned under a laxer threshold or an easier challenge
set be borrowed by a stricter run.

``_verify_report_against_gates`` does not close this: it re-judges the recorded
scores under the current gates, which catches a changed gate contract but still
re-judges measurements taken under the OTHER run's config.

These tests pin the identity comparison directly, because nothing else in the
suite executes ``release check`` at all -- which is why the gap survived.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game_cls.cli.config_tools import _release_identity_mismatches
from game_cls.reports.benchmark import canonical_config_sha256, gate_spec_fingerprint

GATES_STRICT = {"global_fpr_at_decision_threshold": {"max": 0.01}}
GATES_LAX = {"global_fpr_at_decision_threshold": {"max": 0.05}}


def _config(threshold: float, challenge: str | None) -> dict:
    """A config carrying only the fields the identity tuple reads."""
    data: dict = {}
    if challenge is not None:
        data["challenge_video_index"] = challenge
    return {"decision": {"threshold": threshold}, "data": data}


class ReleaseIdentityGateTests(unittest.TestCase):
    def _report(self, directory: Path, payload: dict) -> Path:
        path = directory / "report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _payload(self, *, run_id: str, config: dict, gates: dict) -> dict:
        """The identity fields exactly as ``benchmark evaluate`` writes them."""
        return {
            "run_id": run_id,
            "checkpoint_sha256": "SHARED_WEIGHTS",
            "resolved_config_sha256": canonical_config_sha256(config),
            "challenge_dataset_fingerprint": "",
            "gate_spec_fingerprint": gate_spec_fingerprint(gates),
        }

    def test_matching_identity_is_accepted(self) -> None:
        config = _config(0.99, None)
        with tempfile.TemporaryDirectory() as directory:
            report = self._report(
                Path(directory),
                self._payload(run_id="runA", config=config, gates=GATES_STRICT),
            )
            self.assertEqual(
                _release_identity_mismatches(
                    report, run_id="runA", config=config, gate_metrics=GATES_STRICT
                ),
                [],
            )

    def test_same_weights_different_threshold_cannot_borrow_a_pass(self) -> None:
        """The review's scenario: one checkpoint, two runs, laxer thresholds."""
        lax = _config(0.95, None)
        strict = _config(0.99, None)
        with tempfile.TemporaryDirectory() as directory:
            # Run A earned a PASS at threshold 0.95 under lax gates.
            report = self._report(
                Path(directory),
                self._payload(run_id="runA", config=lax, gates=GATES_LAX),
            )
            # Run B has the SAME checkpoint hash but a stricter identity.
            problems = _release_identity_mismatches(
                report, run_id="runB", config=strict, gate_metrics=GATES_STRICT
            )
        joined = " | ".join(problems)
        self.assertIn("run_id", joined)
        self.assertIn("resolved_config_sha256", joined)
        self.assertIn("gate_spec_fingerprint", joined)

    def test_changed_challenge_set_is_rejected(self) -> None:
        config = _config(0.99, None)
        with tempfile.TemporaryDirectory() as directory:
            payload = self._payload(
                run_id="runA", config=config, gates=GATES_STRICT
            )
            payload["challenge_dataset_fingerprint"] = "CHALLENGE_A"
            report = self._report(Path(directory), payload)
            problems = _release_identity_mismatches(
                report, run_id="runA", config=config, gate_metrics=GATES_STRICT
            )
        # The current side cannot recompute the fingerprint, so the identity is
        # unverifiable and must be refused rather than implicitly accepted.
        self.assertTrue(
            any("challenge_dataset_fingerprint" in problem for problem in problems),
            f"expected a challenge fingerprint problem, got {problems}",
        )

    def test_legacy_report_without_identity_fields_is_refused(self) -> None:
        """A report that cannot prove identity is not a release gate."""
        config = _config(0.99, None)
        with tempfile.TemporaryDirectory() as directory:
            report = self._report(
                Path(directory),
                {"run_id": "runA", "checkpoint_sha256": "SHARED_WEIGHTS"},
            )
            problems = _release_identity_mismatches(
                report, run_id="runA", config=config, gate_metrics=GATES_STRICT
            )
        joined = " | ".join(problems)
        self.assertIn("resolved_config_sha256", joined)
        self.assertIn("gate_spec_fingerprint", joined)

    def test_unreadable_report_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("{not json", encoding="utf-8")
            problems = _release_identity_mismatches(
                path,
                run_id="runA",
                config=_config(0.99, None),
                gate_metrics=GATES_STRICT,
            )
        self.assertEqual(len(problems), 1)
        self.assertIn("unreadable", problems[0])


class CanonicalConfigHashTests(unittest.TestCase):
    """Both sides of the comparison must canonicalize identically (P0-2).

    ``benchmark evaluate`` and ``release check`` compute the config SHA in
    different modules; if they ever diverge, every comparison fails and the
    gate becomes unusable rather than merely weak.
    """

    def test_key_order_does_not_change_the_hash(self) -> None:
        left = {"a": 1, "b": {"c": 2, "d": 3}}
        right = {"b": {"d": 3, "c": 2}, "a": 1}
        self.assertEqual(
            canonical_config_sha256(left), canonical_config_sha256(right)
        )

    def test_value_change_changes_the_hash(self) -> None:
        self.assertNotEqual(
            canonical_config_sha256({"decision": {"threshold": 0.99}}),
            canonical_config_sha256({"decision": {"threshold": 0.95}}),
        )

    def test_evaluate_and_release_check_agree(self) -> None:
        """Pin the shared helper against evaluate's own call path."""
        import inspect

        from game_cls.cli import benchmark as benchmark_cli

        source = inspect.getsource(benchmark_cli.cmd_benchmark_evaluate)
        # evaluate must delegate to the shared helper rather than re-implement
        # the hash, otherwise the two sides can silently drift apart.
        self.assertIn("canonical_config_sha256(config)", source)


if __name__ == "__main__":
    unittest.main()
