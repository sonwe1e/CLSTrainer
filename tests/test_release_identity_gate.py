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

import numpy as np
from PIL import Image

from game_cls.cli.config_tools import _release_identity_mismatches
from game_cls.contract import CONTRACT_VERSION
from game_cls.reports.benchmark import (
    canonical_config_sha256,
    challenge_bundle_fingerprint,
    gate_spec_fingerprint,
)

GATES_STRICT = {"global_fpr_at_decision_threshold": {"op": "<=", "value": 0.01}}
GATES_LAX = {"global_fpr_at_decision_threshold": {"op": "<=", "value": 0.05}}


def _config(threshold: float, challenge: str | None) -> dict:
    """A config carrying only the fields the identity tuple reads."""
    data: dict = {"width": 448, "height": 208, "channels": 3}
    if challenge is not None:
        data["challenge_video_index"] = challenge
    return {"decision": {"threshold": threshold}, "data": data}


def _bundle(directory: Path, *, metadata: bytes = b"meta-a") -> dict:
    """A config pointing at a real on-disk challenge bundle.

    ``benchmark evaluate`` refuses to run without a challenge index at all, so
    a fixture with no challenge files cannot represent a report that really
    exists. Writing the files makes the fingerprint recomputable on both sides,
    which is the state the identity comparison is designed for.
    """
    from game_cls.data.image_spec import ImageSpec
    from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
    from game_cls.data.indexing import write_external_bundle

    source = directory / "source" / "game_a" / "0"
    source.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((208, 448, 3), dtype=np.uint8)).save(
        source / "0100001.png"
    )
    data_cfg = {
        "frame_extensions": [".png"],
        "ignore_directory_prefixes": ["_", "."],
        "ignore_directory_names": ["__pycache__"],
        "ignore_file_globs": [],
    }
    write_external_bundle(
        directory / "source",
        directory,
        ImageSpec(width=448, height=208, channels=3),
        ScanPolicy.from_config(data_cfg),
        DuplicatePolicy(),
        pool="challenge",
        identity_mode="game_label_video",
    )
    frame_index = directory / "frames.parquet"
    video_index = directory / "video_entries.parquet"
    sidecar = directory / "video_metadata.parquet"
    sidecar.write_bytes(metadata)
    config = _config(0.99, str(video_index))
    config["data"]["challenge_index"] = str(frame_index)
    config["data"]["challenge_metadata"] = str(sidecar)
    config["pair"] = {"test_delta": 2}
    config["evaluation"] = {"group_by_negative_subtype": True}
    return config


class ReleaseIdentityGateTests(unittest.TestCase):
    def _report(self, directory: Path, payload: dict) -> Path:
        from game_cls.contract import stamp_payload

        path = directory / "report.json"
        path.write_text(json.dumps(stamp_payload(payload)), encoding="utf-8")
        return path

    def _payload(self, *, run_id: str, config: dict, gates: dict) -> dict:
        """The identity fields exactly as ``benchmark evaluate`` writes them."""
        fingerprint, components = challenge_bundle_fingerprint(config)
        from game_cls.reports.benchmark import benchmark_source_fingerprint

        return {
            "run_id": run_id,
            "checkpoint_sha256": "SHARED_WEIGHTS",
            "resolved_config_sha256": canonical_config_sha256(config),
            "challenge_dataset_fingerprint": fingerprint,
            "challenge_bundle_components": components,
            "gate_spec_fingerprint": gate_spec_fingerprint(gates),
            "contract_version": CONTRACT_VERSION,
            "benchmark_source_sha256": benchmark_source_fingerprint(),
        }

    def test_matching_identity_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
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
            payload = self._payload(run_id="runA", config=config, gates=GATES_STRICT)
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

    def test_report_without_identity_fields_is_refused(self) -> None:
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

    def test_corrupt_challenge_bundle_is_reported_as_identity_problem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _bundle(root)
            report = self._report(
                root,
                self._payload(run_id="runA", config=config, gates=GATES_STRICT),
            )
            (root / "bundle_manifest.json").write_text("{}", encoding="utf-8")
            problems = _release_identity_mismatches(
                report,
                run_id="runA",
                config=config,
                gate_metrics=GATES_STRICT,
            )
        self.assertEqual(len(problems), 1)
        self.assertIn("challenge identity cannot be verified", problems[0])


class ChallengeBundleFingerprintTests(unittest.TestCase):
    """The challenge fingerprint must cover the whole bundle (audit P0-4).

    It used to be ``file_sha256(packed_video_index or video_index)`` -- one
    file. Every test here changes something that provably moves a gated metric
    while leaving that one file byte-identical, which the old definition could
    not see.
    """

    def test_editing_the_metadata_sidecar_changes_the_fingerprint(self) -> None:
        """The audit's exact hole: subtype edits move worst_subtype_fpr."""
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            before, _ = challenge_bundle_fingerprint(config)
            Path(config["data"]["challenge_metadata"]).write_bytes(b"meta-b")
            after, _ = challenge_bundle_fingerprint(config)
        self.assertNotEqual(before, after)

    def test_editing_the_frame_index_changes_the_fingerprint(self) -> None:
        """PNG edits land in the frame index, which the old digest ignored."""
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            video_index = Path(config["data"]["challenge_video_index"])
            before_video = video_index.read_bytes()
            before, _ = challenge_bundle_fingerprint(config)
            Path(config["data"]["challenge_index"]).write_bytes(b"frames-b")
            with self.assertRaises((RuntimeError, ValueError)):
                challenge_bundle_fingerprint(config)
            # The one file the old fingerprint hashed did not move at all.
            self.assertEqual(video_index.read_bytes(), before_video)
        self.assertTrue(before)

    def test_preprocessing_and_geometry_are_part_of_the_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            base, _ = challenge_bundle_fingerprint(config)

            threshold = json.loads(json.dumps(config))
            threshold["decision"]["threshold"] = 0.95
            self.assertNotEqual(base, challenge_bundle_fingerprint(threshold)[0])

            geometry = json.loads(json.dumps(config))
            geometry["data"]["width"] = 224
            self.assertNotEqual(base, challenge_bundle_fingerprint(geometry)[0])

            delta = json.loads(json.dumps(config))
            delta["pair"]["test_delta"] = 3
            self.assertNotEqual(base, challenge_bundle_fingerprint(delta)[0])

    def test_packed_manifest_is_hashed_so_shards_are_covered(self) -> None:
        """The manifest carries shard_sha256, so shards are pinned through it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _bundle(root)
            from game_cls.data.image_spec import ImageSpec
            from game_cls.data.packed_backend import pack_frame_index

            packed = pack_frame_index(
                config["data"]["challenge_index"],
                root / "packed",
                image_spec=ImageSpec(width=448, height=208, channels=3),
                source_video_index=config["data"]["challenge_video_index"],
                audit_path=root / "audit.json",
            )
            manifest = Path(packed).with_name("packed_manifest.json")
            config["data"]["challenge_packed_index"] = str(packed)
            config["data"]["challenge_packed_video_index"] = str(
                Path(packed).with_name("packed_video_entries.parquet")
            )
            before, components = challenge_bundle_fingerprint(config)
            self.assertTrue(components["packed_manifest_sha256"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            shard = manifest.parent / payload["shards"][0]
            shard.write_bytes(shard.read_bytes() + b"corrupt")
            with self.assertRaises(RuntimeError):
                challenge_bundle_fingerprint(config)
        self.assertTrue(before)

    def test_no_resolvable_leg_fails_closed(self) -> None:
        """An unrecomputable identity must not become a valid-looking digest."""
        digest, _ = challenge_bundle_fingerprint(_config(0.99, None))
        self.assertEqual(digest, "")

    def test_writer_and_reader_agree_on_an_unchanged_bundle(self) -> None:
        """Both sides must move together or every comparison fails (P0-4)."""
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            report = ReleaseIdentityGateTests()._report(
                Path(directory),
                ReleaseIdentityGateTests()._payload(
                    run_id="runA", config=config, gates=GATES_STRICT
                ),
            )
            problems = _release_identity_mismatches(
                report, run_id="runA", config=config, gate_metrics=GATES_STRICT
            )
        self.assertEqual(problems, [])

    def test_mismatch_names_the_leg_that_moved(self) -> None:
        """An operator must not be left diffing two opaque digests."""
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            report = ReleaseIdentityGateTests()._report(
                Path(directory),
                ReleaseIdentityGateTests()._payload(
                    run_id="runA", config=config, gates=GATES_STRICT
                ),
            )
            Path(config["data"]["challenge_metadata"]).write_bytes(b"meta-b")
            problems = _release_identity_mismatches(
                report, run_id="runA", config=config, gate_metrics=GATES_STRICT
            )
        joined = " | ".join(problems)
        self.assertIn("challenge_dataset_fingerprint", joined)
        self.assertIn("challenge_metadata_sha256", joined)

    def test_report_predating_the_widened_fingerprint_is_refused(self) -> None:
        """A narrow digest must not be accepted as a component identity."""
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            payload = ReleaseIdentityGateTests()._payload(
                run_id="runA", config=config, gates=GATES_STRICT
            )
            # A malformed report: a single-file digest with no components block.
            from game_cls.reports.benchmark import file_sha256

            payload["challenge_dataset_fingerprint"] = file_sha256(
                config["data"]["challenge_video_index"]
            )
            payload.pop("challenge_bundle_components", None)
            report = ReleaseIdentityGateTests()._report(Path(directory), payload)
            problems = _release_identity_mismatches(
                report, run_id="runA", config=config, gate_metrics=GATES_STRICT
            )
        joined = " | ".join(problems)
        self.assertIn("challenge_dataset_fingerprint", joined)
        self.assertIn("predates the widened challenge fingerprint", joined)

    def test_covers_content_reports_whether_pixels_are_covered(self) -> None:
        """A bundle indexed without content hashes is visibly weaker."""
        from game_cls.data.indexing import write_parquet
        from game_cls.data.records import FrameRecord

        def _frame(content: str) -> FrameRecord:
            return FrameRecord(
                sample_id="challenge:g:0:v:00001",
                split="challenge",
                game="g",
                label=0,
                video_id="v",
                frame_id=1,
                path="g/0/v00001.png",
                width=448,
                height=208,
                channels=3,
                file_size=1234,
                content_sha256=content,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _bundle(root)
            index = Path(config["data"]["challenge_index"])

            write_parquet([_frame("")], index)
            with self.assertRaises((RuntimeError, ValueError)):
                challenge_bundle_fingerprint(config)

    def test_current_png_bytes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _bundle(Path(directory))
            frame_path = next((Path(directory) / "source").rglob("*.png"))
            Image.fromarray(np.ones((208, 448, 3), dtype=np.uint8)).save(frame_path)
            with self.assertRaisesRegex(ValueError, "changed after indexing"):
                challenge_bundle_fingerprint(config)


class CanonicalConfigHashTests(unittest.TestCase):
    """Both sides of the comparison must canonicalize identically (P0-2).

    ``benchmark evaluate`` and ``release check`` compute the config SHA in
    different modules; if they ever diverge, every comparison fails and the
    gate becomes unusable rather than merely weak.
    """

    def test_key_order_does_not_change_the_hash(self) -> None:
        left = {"a": 1, "b": {"c": 2, "d": 3}}
        right = {"b": {"d": 3, "c": 2}, "a": 1}
        self.assertEqual(canonical_config_sha256(left), canonical_config_sha256(right))

    def test_value_change_changes_the_hash(self) -> None:
        self.assertNotEqual(
            canonical_config_sha256({"decision": {"threshold": 0.99}}),
            canonical_config_sha256({"decision": {"threshold": 0.95}}),
        )

    def test_evaluate_and_release_check_agree(self) -> None:
        """Pin the shared helper against evaluate's own call path."""
        import inspect

        import game_cls.cli.benchmark as benchmark_cli

        source = inspect.getsource(benchmark_cli.cmd_benchmark_evaluate)
        # evaluate must delegate to the shared helper rather than re-implement
        # the hash, otherwise the two sides can silently drift apart.
        self.assertIn("canonical_config_sha256(config)", source)


if __name__ == "__main__":
    unittest.main()
