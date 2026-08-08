from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
from game_cls.data.indexing import (
    read_frame_parquet,
    scan_split,
    validate_audit,
    write_index_bundle,
    write_split_bundle,
)
from game_cls.data.splitter import (
    SPLIT_ALGORITHM_VERSION,
    load_split_manifest,
    source_video_uid,
)


def write_png_header(path: Path, width: int = 448, height: int = 208) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + bytes([8, 2])
    )


def write_unique_png(path: Path, width: int = 448, height: int = 208) -> None:
    """A scan-valid PNG whose file bytes are unique across the dataset.

    The 26-byte header carries the configured dimensions, and a
    per-path suffix makes the SHA-256 content hash unique. Without the
    suffix every temp frame would be byte-identical and the content
    duplicate audit would flag cross-label/cross-split collisions.
    """
    write_png_header(path, width, height)
    with path.open("ab") as stream:
        stream.write(str(path.resolve()).encode("utf-8"))


def write_video_frames(
    root: Path,
    game: str,
    label: int,
    video_id: str,
    frame_ids: list[int],
) -> None:
    for frame_id in frame_ids:
        write_unique_png(root / game / str(label) / f"{video_id}{frame_id:05d}.png")


def scan_policy(**overrides) -> ScanPolicy:
    config = {
        "frame_extensions": [".png"],
        "ignore_directory_prefixes": ["_", "."],
        "ignore_directory_names": [
            "__pycache__",
            "cache",
            "caches",
            "tmp",
            "temp",
        ],
        "ignore_file_globs": ["*.tmp", "*.part", "*.log"],
        "unexpected_nested_directory_severity": "warning",
        **overrides,
    }
    return ScanPolicy.from_config(config)


class ImageSpecAndScanTests(unittest.TestCase):
    def test_validates_training_pair_batch_shape(self) -> None:
        import torch

        spec = ImageSpec(width=448, height=208, channels=3)
        images = torch.zeros((2, 2, 3, 208, 448))
        spec.validate_pair_batch_shape(images.shape)
        with self.assertRaisesRegex(RuntimeError, "Training image shape mismatch"):
            spec.validate_pair_batch_shape((2, 2, 3, 448, 208))

    def test_uses_configured_landscape_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png_header(root / "game_A" / "0" / "0100001.png")
            write_png_header(
                root / "game_A" / "0" / "0100002.png",
                width=208,
                height=448,
            )
            result = scan_split(
                root,
                "train",
                ImageSpec(width=448, height=208, channels=3),
                scan_policy=scan_policy(),
                compute_content_hash=False,
            )
            self.assertEqual(len(result.frames), 1)
            self.assertEqual(
                (result.frames[0].width, result.frames[0].height),
                (448, 208),
            )
            self.assertEqual(
                result.findings.errors[0]["kind"],
                "unexpected_dimensions",
            )

    def test_different_configured_dimensions_require_no_code_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png_header(
                root / "game_A" / "0" / "0100001.png",
                width=320,
                height=120,
            )
            result = scan_split(
                root,
                "train",
                ImageSpec(width=320, height=120, channels=3),
                scan_policy=scan_policy(),
                compute_content_hash=False,
            )
            self.assertEqual(len(result.frames), 1)
            self.assertFalse(result.findings.errors)

    def test_structured_scan_ignores_auxiliary_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png_header(root / "game_A" / "0" / "0100001.png")
            (root / "game_A" / "0" / "video01.mp4").write_bytes(b"video")
            (root / "game_A" / "0" / "meta.json").write_text("{}")
            for index in range(25):
                (root / "game_A" / "0" / f"aux_{index:02d}.json").write_text("{}")
            write_png_header(root / "game_A" / "0" / "_cache" / "thumb.png")
            write_png_header(root / "game_A" / "_cache" / "foo.png")
            write_png_header(root / "game_A" / "0" / "bad_name.png")
            write_png_header(root / "game_A" / "2" / "0100001.png")
            write_png_header(root / "game_A" / "0" / "random_dir" / "file.png")

            result = scan_split(
                root,
                "train",
                ImageSpec(width=448, height=208, channels=3),
                scan_policy=scan_policy(),
                compute_content_hash=False,
            )

            self.assertEqual(
                [Path(frame.path).name for frame in result.frames],
                ["0100001.png"],
            )
            self.assertEqual(
                {item["kind"] for item in result.findings.errors},
                {"invalid_filename", "invalid_label_directory"},
            )
            self.assertEqual(
                [item["kind"] for item in result.findings.warnings],
                ["unexpected_nested_directory"],
            )
            self.assertEqual(result.findings.ignored_counts["non_frame_extension"], 27)
            self.assertEqual(
                len(result.findings.ignored_examples["non_frame_extension"]),
                20,
            )
            self.assertEqual(result.findings.ignored_counts["ignored_directory"], 2)

            strict_nested = scan_split(
                root,
                "train",
                ImageSpec(width=448, height=208, channels=3),
                scan_policy=scan_policy(unexpected_nested_directory_severity="error"),
                compute_content_hash=False,
            )
            self.assertIn(
                "unexpected_nested_directory",
                {item["kind"] for item in strict_nested.findings.errors},
            )


@unittest.skipIf(pq is None, "pyarrow is not installed in the current interpreter")
class IndexBundleTests(unittest.TestCase):
    def test_writes_frame_video_indexes_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                for frame_id in (1, 2, 4):
                    write_png_header(
                        root / split / "game_A" / "0" / f"01{frame_id:05d}.png"
                    )
            output = root / "indexes"
            audit = write_index_bundle(
                root / "train",
                root / "test",
                output,
                ImageSpec(width=448, height=208, channels=3),
                scan_policy(),
                DuplicatePolicy(),
            )
            self.assertEqual(audit["audit_format_version"], 3)
            self.assertEqual(
                audit["expected"],
                {"width": 448, "height": 208, "channels": 3},
            )
            self.assertEqual(audit["splits"]["train"]["valid_pairs"]["2"], 1)
            self.assertEqual(
                audit["splits"]["train"]["valid_pairs_by_game_label_delta"][1],
                {
                    "game": "game_A",
                    "label": 0,
                    "delta": 2,
                    "count": 1,
                },
            )
            self.assertEqual(pq.read_table(output / "train_frames.parquet").num_rows, 3)
            videos = pq.read_table(output / "train_videos.parquet").to_pylist()
            self.assertEqual(videos[0]["valid_pair_count_delta2"], 1)
            self.assertTrue((output / "audit.json").is_file())
            self.assertTrue((output / "train_video_entries.parquet").is_file())
            video_entries = pq.read_table(
                output / "train_video_entries.parquet"
            ).to_pylist()
            self.assertEqual(video_entries[0]["frame_ids"], [1, 2, 4])
            self.assertEqual(video_entries[0]["valid_starts_delta2"], [1])
            self.assertIsNone(video_entries[0]["frame_paths"])
            self.assertTrue(video_entries[0]["video_directory"])
            self.assertTrue(audit["duplicates"]["warnings"])

    def test_missing_video_index_raises_actionable_error(self) -> None:
        """A missing val index must explain how to regenerate it instead of
        leaking a raw pyarrow FileNotFoundError."""
        from game_cls.data.video_index import read_video_entries_parquet

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "val_video_entries.parquet"
            with self.assertRaises(FileNotFoundError) as ctx:
                read_video_entries_parquet(missing, (2,))
            message = str(ctx.exception)
            self.assertIn("Video-level index is missing", message)
            self.assertIn("--val-root", message)
            self.assertIn("build_index.py", message)

    def test_three_split_bundle_and_source_uid_leakage(self) -> None:
        """A source video spanning train/val/test must be detected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Same video id '01' in train, val and test: leakage.
            for split in ("train", "val", "test"):
                write_png_header(root / split / "game_A" / "0" / "0100001.png")
            output = root / "indexes"
            audit = write_index_bundle(
                root / "train",
                root / "test",
                output,
                ImageSpec(width=448, height=208, channels=3),
                scan_policy(),
                DuplicatePolicy(),
                val_root=root / "val",
            )
            self.assertEqual(
                audit["leakage"]["source_video_uid_overlap"]["train__test"],
                ["game_A::01"],
            )
            self.assertEqual(
                audit["leakage"]["source_video_uid_overlap"]["train__val"],
                ["game_A::01"],
            )
            with self.assertRaisesRegex(RuntimeError, "source videos span"):
                from game_cls.data.indexing import validate_audit

                validate_audit(
                    audit,
                    image_spec=ImageSpec(width=448, height=208, channels=3),
                    duplicate_policy=DuplicatePolicy(),
                    require_content_hash=False,
                )

    def test_three_split_bundle_with_namespaces_separates_test(self) -> None:
        """A distinct test namespace clears train/test id coincidence while
        the same-namespace train/val overlap stays fatal (train/val always
        share one pool)."""
        namespaces_by_split = {
            "train": "train_pool",
            "val": "train_pool",
            "test": "heldout_pool",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                write_png_header(root / split / "game_A" / "0" / "0100001.png")
            output = root / "indexes"
            audit = write_index_bundle(
                root / "train",
                root / "test",
                output,
                ImageSpec(width=448, height=208, channels=3),
                scan_policy(),
                DuplicatePolicy(),
                val_root=root / "val",
                namespaces_by_split=namespaces_by_split,
            )
            overlap = audit["leakage"]["source_video_uid_overlap"]
            # train/val share the train_pool namespace: the real overlap of
            # the same source video stays detectable.
            self.assertEqual(overlap["train__val"], ["train_pool::game_A::01"])
            # test lives in a distinct pool: coincidental ids are not leakage.
            self.assertEqual(overlap["train__test"], [])
            self.assertEqual(overlap["val__test"], [])
            self.assertEqual(
                audit["leakage"]["source_identity_namespaces"],
                dict(sorted(namespaces_by_split.items())),
            )
            with self.assertRaisesRegex(RuntimeError, "train/val"):
                from game_cls.data.indexing import validate_audit

                validate_audit(
                    audit,
                    image_spec=ImageSpec(width=448, height=208, channels=3),
                    duplicate_policy=DuplicatePolicy(),
                    require_content_hash=False,
                    namespaces_by_split=namespaces_by_split,
                )


@unittest.skipIf(pq is None, "pyarrow is not installed in the current interpreter")
class SplitBundleTests(unittest.TestCase):
    """write_split_bundle: derive train/val from a single train_all root."""

    def _data_config(self) -> dict:
        return {
            "width": 448,
            "height": 208,
            "channels": 3,
            "frame_extensions": [".png"],
            "ignore_directory_prefixes": ["_", "."],
            "ignore_directory_names": [
                "__pycache__",
                "cache",
                "caches",
                "tmp",
                "temp",
            ],
            "ignore_file_globs": ["*.tmp", "*.part", "*.log"],
            "unexpected_nested_directory_severity": "warning",
        }

    def _split_config(self) -> dict:
        return {
            "mode": "from_train",
            "val_ratio": 0.2,
            "seed": 20260728,
            "group_key": "source_video_uid",
            "stratify_by": ["game", "label"],
            "balance_by": "legal_pair_count",
            "target_delta": 2,
            "manifest": "split_manifest.parquet",
            "on_new_groups": "error",
            "small_stratum_policy": "error",
        }

    def _write_roots(self, root: Path) -> None:
        # video ids must be unique per game (source_video_uid is
        # label-independent), so a per-game counter spans both labels.
        train_all = root / "train_all"
        for game in ("game_a", "game_b", "game_c"):
            for video in range(8):
                label = video % 2
                write_video_frames(
                    train_all,
                    game,
                    label,
                    f"{video + 1:02d}",
                    list(range(1, 13)),
                )
        test_root = root / "test"
        for label in (0, 1):
            write_video_frames(test_root, "game_d", label, "01", [1, 2, 3])

    def test_write_split_bundle_produces_full_artifacts(self) -> None:
        data_config = self._data_config()
        image_spec = ImageSpec.from_config(data_config)
        scan_policy = ScanPolicy.from_config(data_config)
        duplicate_policy = DuplicatePolicy.from_config(data_config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_roots(root)
            output = root / "indexes"
            audit = write_split_bundle(
                root / "train_all",
                root / "test",
                output,
                image_spec,
                scan_policy,
                duplicate_policy,
                split_config=self._split_config(),
            )

            for split in ("train", "val", "test"):
                self.assertTrue(
                    (output / f"{split}_frames.parquet").is_file(),
                    f"missing {split}_frames.parquet",
                )
                self.assertTrue((output / f"{split}_videos.parquet").is_file())
                self.assertTrue((output / f"{split}_video_entries.parquet").is_file())
            self.assertTrue((output / "split_manifest.parquet").is_file())
            self.assertTrue((output / "split_summary.json").is_file())
            self.assertTrue((output / "audit.json").is_file())

            # Every source video lives in exactly one split: no uid may span
            # any pair of train/val/test.
            for pair_key, uids in audit["leakage"]["source_video_uid_overlap"].items():
                self.assertEqual(
                    uids,
                    [],
                    f"source videos span the {pair_key} splits: {uids}",
                )

            # The manifest assignment agrees with the emitted frame rows and
            # every frame carries its split's label.
            manifest = load_split_manifest(output / "split_manifest.parquet")
            self.assertEqual(manifest["split_seed"], 20260728)
            self.assertEqual(
                manifest["split_algorithm_version"], SPLIT_ALGORITHM_VERSION
            )
            for split in ("train", "val"):
                frames = read_frame_parquet(output / f"{split}_frames.parquet")
                self.assertTrue(frames, f"{split} has no frames")
                for frame in frames:
                    self.assertEqual(frame.split, split)
                    self.assertEqual(
                        manifest["assignment"][
                            source_video_uid(frame.game, frame.video_id)
                        ],
                        split,
                    )

            # Both derived splits and the independent test split carry legal
            # delta=2 pairs, and the strict audit gate passes with the exact
            # policies that built the bundle.
            self.assertGreater(audit["splits"]["val"]["valid_pairs"]["2"], 0)
            self.assertGreater(audit["splits"]["test"]["valid_pairs"]["2"], 0)
            self.assertIn("split", audit)
            self.assertEqual(
                audit["split"]["split_algorithm_version"],
                SPLIT_ALGORITHM_VERSION,
            )
            self.assertFalse(audit["split"]["manifest_reused"])
            # The summary names the ratio after the delta that drove the
            # balancing, so a delta!=2 split is not misreported.
            self.assertEqual(audit["split"]["target_delta"], 2)
            self.assertEqual(
                audit["split"]["val_ratio_achieved"],
                audit["split"]["val_ratio_achieved_delta2"],
            )
            validate_audit(
                audit,
                image_spec=image_spec,
                scan_policy=scan_policy,
                duplicate_policy=duplicate_policy,
                require_test_delta=2,
            )

    def test_write_split_bundle_rejects_bad_split_config(self) -> None:
        data_config = self._data_config()
        image_spec = ImageSpec.from_config(data_config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_roots(root)
            with self.assertRaisesRegex(ValueError, "mode must be 'from_train'"):
                write_split_bundle(
                    root / "train_all",
                    root / "test",
                    root / "indexes",
                    image_spec,
                    ScanPolicy.from_config(data_config),
                    DuplicatePolicy.from_config(data_config),
                    split_config={**self._split_config(), "mode": "off"},
                )


class SplitBundleTransactionTests(SplitBundleTests):
    """The bundle must be provably ONE generation (audit P0-9).

    ``write_split_bundle`` used to overwrite artifacts in place, in order:
    train, val, test, video indexes, summary, audit. A crash partway through,
    over a directory that already held a complete older bundle, left
    ``train=NEW, test=OLD, audit=OLD`` with *every file present* -- so an
    existence check reported nothing missing and training silently proceeded on
    a mixed split. Existence checks cannot close that, because existence is
    exactly what the mixed state satisfies.
    """

    def _prepare(self, root: Path) -> Path:
        from game_cls.data.indexing import write_split_bundle

        data_config = self._data_config()
        self._write_roots(root)
        output = root / "indexes"
        write_split_bundle(
            root / "train_all",
            root / "test",
            output,
            ImageSpec.from_config(data_config),
            ScanPolicy.from_config(data_config),
            DuplicatePolicy.from_config(data_config),
            split_config=self._split_config(),
        )
        return output

    def test_prepare_commits_a_bundle_manifest_that_verifies(self) -> None:
        from game_cls.data.indexing import (
            BUNDLE_ARTIFACT_NAMES,
            read_bundle_manifest,
            verify_split_bundle,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            manifest = read_bundle_manifest(output)
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertTrue(manifest["bundle_id"])
            # Every published artifact is committed.
            for name in BUNDLE_ARTIFACT_NAMES:
                self.assertIn(name, manifest["artifacts"], name)
            # The split manifest is deliberately NOT hash-committed:
            # resolve_split mutates it before the parquets are staged, so
            # hashing it would report MIXED GENERATION on a bundle whose
            # train/val/test are still one internally consistent set.
            self.assertNotIn("split_manifest.parquet", manifest["artifacts"])
            self.assertIsNotNone(verify_split_bundle(output))

    def test_extending_the_split_manifest_alone_is_not_a_mixed_generation(self) -> None:
        """The false positive this relaxation exists to prevent.

        resolve_split may extend the manifest before the parquets are staged. A
        crash in that window leaves the manifest moved and the parquets intact,
        which is not a mixed bundle and must not be reported as one.
        """
        from game_cls.data.indexing import verify_split_bundle

        with tempfile.TemporaryDirectory() as directory:
            output = self._prepare(Path(directory))
            manifest_file = output / "split_manifest.parquet"
            manifest_file.write_bytes(manifest_file.read_bytes() + b"\x00extended")
            # Still one generation as far as the split artifacts go.
            self.assertIsNotNone(verify_split_bundle(output))

    def test_the_generation_is_stamped_into_audit_and_summary(self) -> None:
        import json

        from game_cls.data.indexing import read_bundle_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            manifest = read_bundle_manifest(output)
            assert manifest is not None
            bundle_id = manifest["bundle_id"]
            for name in ("audit.json", "split_summary.json"):
                payload = json.loads((output / name).read_text(encoding="utf-8"))
                self.assertEqual(payload.get("bundle_id"), bundle_id, name)

    def test_mixed_generation_is_refused_though_every_file_exists(self) -> None:
        """The audit's exact scenario: test_* left over from an older prepare."""
        from game_cls.data.indexing import SplitBundleError, verify_split_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            stale = (output / "test_frames.parquet").read_bytes() + b"older-generation"
            (output / "test_frames.parquet").write_bytes(stale)
            # Nothing is missing -- that is the whole point of the bug.
            for name in ("train_frames.parquet", "test_frames.parquet", "audit.json"):
                self.assertTrue((output / name).is_file())
            with self.assertRaises(SplitBundleError) as caught:
                verify_split_bundle(output)
            message = str(caught.exception)
            self.assertIn("MIXED GENERATION", message)
            # The operator needs to know WHICH artifact disagrees.
            self.assertIn("test_frames.parquet", message)

    def test_an_audit_swapped_from_another_bundle_is_refused(self) -> None:
        from game_cls.data.indexing import (
            SplitBundleError,
            verify_split_bundle,
            write_bundle_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            # Re-commit so the hashes match, but leave audit.json stamped with a
            # different generation: the hash check alone would pass.
            import json

            payload = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            payload["bundle_id"] = "a-different-generation"
            (output / "audit.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_bundle_manifest(output, "the-committed-generation")
            with self.assertRaises(SplitBundleError) as caught:
                verify_split_bundle(output)
            message = str(caught.exception)
            self.assertIn("MIXED GENERATION", message)
            self.assertIn("audit.json", message)

    def test_a_missing_artifact_is_reported_as_incomplete_not_mixed(self) -> None:
        """Three distinct failures need three distinct operator actions."""
        from game_cls.data.indexing import SplitBundleError, verify_split_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            (output / "val_video_entries.parquet").unlink()
            with self.assertRaises(SplitBundleError) as caught:
                verify_split_bundle(output)
            message = str(caught.exception)
            self.assertIn("incomplete", message)
            self.assertIn("val_video_entries.parquet", message)

    def test_an_unsealed_directory_is_refused_but_can_be_tolerated(self) -> None:
        from game_cls.data.indexing import (
            BUNDLE_MANIFEST_FILENAME,
            SplitBundleError,
            verify_split_bundle,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            (output / BUNDLE_MANIFEST_FILENAME).unlink()
            with self.assertRaises(SplitBundleError) as caught:
                verify_split_bundle(output)
            message = str(caught.exception)
            # Both ways forward must be named, since only the operator knows
            # whether a legacy directory is really one generation.
            self.assertIn("dataset prepare", message)
            self.assertIn("dataset seal", message)
            # Callers that only want "was anything built" opt out explicitly.
            self.assertIsNone(verify_split_bundle(output, require_manifest=False))

    def test_no_staging_directory_survives_a_successful_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            self.assertEqual(
                [path.name for path in output.iterdir() if path.name.startswith(".staging")],
                [],
            )

    def test_re_preparing_commits_a_new_generation_that_still_verifies(self) -> None:
        from game_cls.data.indexing import (
            read_bundle_manifest,
            verify_split_bundle,
            write_split_bundle,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self._prepare(root)
            first = read_bundle_manifest(output)
            assert first is not None
            data_config = self._data_config()
            write_split_bundle(
                root / "train_all",
                root / "test",
                output,
                ImageSpec.from_config(data_config),
                ScanPolicy.from_config(data_config),
                DuplicatePolicy.from_config(data_config),
                split_config=self._split_config(),
            )
            second = read_bundle_manifest(output)
            assert second is not None
            self.assertNotEqual(first["bundle_id"], second["bundle_id"])
            # A re-prepare must leave a bundle that still verifies, not one that
            # trips its own mixed-generation check.
            self.assertIsNotNone(verify_split_bundle(output))


if __name__ == "__main__":
    unittest.main()
