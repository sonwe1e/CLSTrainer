"""Source video identity contract tests (step7 §六, §八).

Covers the explicit ``source_video_identity.mode`` switch, the source
identity precheck report and the identity-mode-aware split bundle:
``game_video`` (default) treats video_id as unique per game while
``game_label_video`` treats it as unique per (game, label).
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from game_cls.config import load_config
from game_cls.config_schema import ConfigSchemaError, finalize_config
from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
from game_cls.data.indexing import write_split_bundle
from game_cls.data.records import FrameRecord
from game_cls.data.splitter import (
    SOURCE_IDENTITY_MODES,
    format_source_identity_precheck,
    load_split_manifest,
    source_identity_precheck,
    source_video_uid,
)


def _frames(
    game: str,
    label: int,
    video_id: str,
    count: int,
    root: str = "/data",
):
    return [
        FrameRecord(
            sample_id=f"train:{game}:{label}:{video_id}:{i:05d}",
            split="train",
            game=game,
            label=label,
            video_id=video_id,
            frame_id=i,
            path=f"{root}/{game}/{label}/{video_id}{i:05d}.png",
            width=448,
            height=208,
            channels=3,
            file_size=1,
            content_sha256=f"sha-{game}-{label}-{video_id}-{i}",
        )
        for i in range(count)
    ]


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
    """A scan-valid PNG whose file bytes are unique across the dataset."""
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


class SourceIdentityModeTests(unittest.TestCase):
    def test_uid_defaults_to_game_video(self) -> None:
        self.assertEqual(source_video_uid("g", "01"), "g::01")
        self.assertEqual(
            source_video_uid("g", "01", 0, mode="game_video"), "g::01"
        )

    def test_game_label_video_mode_namespaced_by_label(self) -> None:
        self.assertEqual(
            source_video_uid("g", "01", 0, mode="game_label_video"), "g::0::01"
        )
        self.assertEqual(
            source_video_uid("g", "01", 1, mode="game_label_video"), "g::1::01"
        )
        self.assertNotEqual(
            source_video_uid("g", "01", 0, mode="game_label_video"),
            source_video_uid("g", "01", 1, mode="game_label_video"),
        )

    def test_source_identity_modes_are_defined(self) -> None:
        self.assertIn("game_video", SOURCE_IDENTITY_MODES)
        self.assertIn("game_label_video", SOURCE_IDENTITY_MODES)


class SourceIdentityPrecheckTests(unittest.TestCase):
    def test_precheck_counts_single_and_mixed_label(self) -> None:
        frames = (
            _frames("MC", 0, "01", 8)
            + _frames("MC", 1, "01", 8)  # mixed: MC::01 spans both labels
            + _frames("MC", 0, "02", 8)  # single label 0
            + _frames("MC", 1, "03", 8)  # single label 1
        )
        report = source_identity_precheck(frames, identity_mode="game_video")
        self.assertEqual(report["source_video_count"], 3)
        self.assertEqual(report["single_label_videos"], 2)
        self.assertEqual(report["mixed_label_videos"], 1)
        # delta=2 pairs per label: 6 per 8-frame run, two runs per label.
        self.assertEqual(report["pair_counts_label0"][2], 12)
        self.assertEqual(report["pair_counts_label1"][2], 12)

    def test_precheck_game_label_video_reports_no_mixed(self) -> None:
        # Same video_id under both labels: two independent source videos.
        frames = _frames("MC", 0, "01", 8) + _frames("MC", 1, "01", 8)
        report = source_identity_precheck(frames, identity_mode="game_label_video")
        self.assertEqual(report["source_video_count"], 2)
        self.assertEqual(report["single_label_videos"], 2)
        self.assertEqual(report["mixed_label_videos"], 0)

    def test_format_precheck_reports_support_and_atomicity(self) -> None:
        frames = (
            _frames("MC", 0, "01", 8)
            + _frames("MC", 1, "01", 8)
            + _frames("MC", 0, "02", 8)
            + _frames("MC", 1, "02", 8)
        )
        report = source_identity_precheck(frames, identity_mode="game_video")
        text = format_source_identity_precheck(report)
        self.assertIn("Supported: yes", text)
        self.assertIn("Atomic split enforced: yes", text)


class SourceIdentityConfigTests(unittest.TestCase):
    def test_config_defaults_to_game_video(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        self.assertEqual(
            config["data"]["source_video_identity"]["mode"], "game_video"
        )

    def test_config_unknown_mode_is_rejected(self) -> None:
        config = load_config("configs/cuda_debug.yaml")
        config["data"]["source_video_identity"]["mode"] = "game_label"
        with self.assertRaisesRegex(ConfigSchemaError, "source_video_identity"):
            finalize_config(config)


class SourceIdentityParityTests(unittest.TestCase):
    def test_indexing_reuses_splitter_source_video_uid(self) -> None:
        from game_cls.data import indexing, splitter

        self.assertIs(indexing.source_video_uid, splitter.source_video_uid)


@unittest.skipIf(pq is None, "pyarrow is not installed in the current interpreter")
class SourceIdentityBundleTests(unittest.TestCase):
    """write_split_bundle records the identity mode on the manifest."""

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
            "source_identity_mode": "game_label_video",
        }

    def test_write_split_bundle_records_game_label_video_mode(self) -> None:
        data_config = self._data_config()
        image_spec = ImageSpec.from_config(data_config)
        scan_policy = ScanPolicy.from_config(data_config)
        duplicate_policy = DuplicatePolicy.from_config(data_config)
        split_config = self._split_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_all = root / "train_all"
            # Identical video ids under both labels are distinct source
            # videos in game_label_video mode; each (game, label) stratum
            # keeps three videos so the split is well-formed.
            for label in (0, 1):
                for video_id in ("01", "02", "03"):
                    write_video_frames(
                        train_all, "game_a", label, video_id, list(range(1, 13))
                    )
            test_root = root / "test"
            for label in (0, 1):
                write_video_frames(
                    test_root, "game_d", label, "01", [1, 2, 3]
                )
            output = root / "indexes"
            write_split_bundle(
                train_all,
                test_root,
                output,
                image_spec,
                scan_policy,
                duplicate_policy,
                split_config=split_config,
                identity_mode="game_label_video",
            )
            manifest = load_split_manifest(output / "split_manifest.parquet")
            self.assertEqual(
                manifest["split_source_identity_mode"], "game_label_video"
            )

    def test_game_label_video_bundle_passes_identity_aware_strict_audit(self) -> None:
        # step7 regression: validate_audit_file was never handed the identity
        # mode, so a bundle built under game_label_video always failed the
        # strict gate with "identity mode does not match configuration".
        from game_cls.data.indexing import validate_audit_file

        data_config = self._data_config()
        image_spec = ImageSpec.from_config(data_config)
        scan_policy = ScanPolicy.from_config(data_config)
        duplicate_policy = DuplicatePolicy.from_config(data_config)
        split_config = self._split_config()  # records source_identity_mode
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_all = root / "train_all"
            for label in (0, 1):
                for video_id in ("01", "02", "03"):
                    write_video_frames(
                        train_all, "game_a", label, video_id, list(range(1, 13))
                    )
            test_root = root / "test"
            for label in (0, 1):
                write_video_frames(
                    test_root, "game_d", label, "01", [1, 2, 3]
                )
            output = root / "indexes"
            write_split_bundle(
                train_all,
                test_root,
                output,
                image_spec,
                scan_policy,
                duplicate_policy,
                split_config=split_config,
                identity_mode="game_label_video",
            )
            # With the identity mode threaded, the strict gate passes.
            validate_audit_file(
                output / "audit.json",
                image_spec=image_spec,
                scan_policy=scan_policy,
                duplicate_policy=duplicate_policy,
                require_test_delta=2,
                identity_mode="game_label_video",
            )

    def test_write_split_bundle_rejects_conflicting_identity_mode(self) -> None:
        # A split_config that records source_identity_mode must not disagree
        # silently with the explicit identity_mode kwarg.
        data_config = self._data_config()
        image_spec = ImageSpec.from_config(data_config)
        scan_policy = ScanPolicy.from_config(data_config)
        duplicate_policy = DuplicatePolicy.from_config(data_config)
        split_config = self._split_config()  # source_identity_mode=game_label_video
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_all = root / "train_all"
            for label in (0, 1):
                for video_id in ("01", "02", "03"):
                    write_video_frames(
                        train_all, "game_a", label, video_id, list(range(1, 13))
                    )
            test_root = root / "test"
            for label in (0, 1):
                write_video_frames(
                    test_root, "game_d", label, "01", [1, 2, 3]
                )
            with self.assertRaisesRegex(ValueError, "disagree"):
                write_split_bundle(
                    train_all,
                    test_root,
                    root / "indexes",
                    image_spec,
                    scan_policy,
                    duplicate_policy,
                    split_config=split_config,
                    identity_mode="game_video",
                )


if __name__ == "__main__":
    unittest.main()
