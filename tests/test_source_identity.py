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
from game_cls.config_schema import (
    ConfigSchemaError,
    finalize_config,
    resolve_source_identity_namespaces,
)
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


class StableAndContentVersionIdentityTests(unittest.TestCase):
    """Stable physical identity is independent from the content version.

    The source identity namespaces are audit-boundary-only, so coincidentally
    equal train/test local numbering must never collapse at the metadata layer.
    The canonical ``source_video_uid`` therefore carries a content signature
    (frame count + ordered per-frame hashes) that distinguishes distinct pools
    without touching the namespace dimension.
    """

    def _content_frames(self, game, label, video_id, hashes):
        return [
            FrameRecord(
                sample_id=f"x:{game}:{label}:{video_id}:{i:05d}",
                split="train",
                game=game,
                label=label,
                video_id=video_id,
                frame_id=i,
                path=f"/d/{game}/{label}/{video_id}{i:05d}.png",
                width=448,
                height=208,
                channels=3,
                file_size=1,
                content_sha256=content_hash,
            )
            for i, content_hash in enumerate(hashes, start=1)
        ]

    def test_content_anchored_uid_distinguishes_coincidental_numbering(self) -> None:
        from game_cls.data.video_index import build_video_entries

        train_pool = self._content_frames("MC", 0, "01", ("a", "b", "c"))
        test_pool = self._content_frames("MC", 0, "01", ("d", "e", "f"))
        train_video = build_video_entries(train_pool)[0]
        test_video = build_video_entries(test_pool)[0]
        self.assertEqual(train_video.stable_source_id, test_video.stable_source_id)
        self.assertNotEqual(
            train_video.content_version_id, test_video.content_version_id
        )
        self.assertNotEqual(train_video.source_version_id, test_video.source_version_id)

    def test_content_id_is_deterministic(self) -> None:
        from game_cls.data.video_index import build_video_entries

        frames = self._content_frames("MC", 0, "01", ("a", "b", "c"))
        first = build_video_entries(frames)[0].source_version_id
        second = build_video_entries(frames)[0].source_version_id
        self.assertEqual(first, second)
        # Identical content in both pools is the SAME video: uid must collide
        # so the strict audit's SHA-256 layer can catch a real leak.
        same = build_video_entries(self._content_frames("MC", 0, "01", ("a", "b", "c")))
        self.assertEqual(same[0].source_version_id, first)

    def test_sidecar_join_never_bleeds_across_pools(self) -> None:
        from game_cls.data.sidecar import apply_sidecar
        from game_cls.data.video_index import build_video_entries

        train_pool = self._content_frames("MC", 0, "01", ("a", "b", "c"))
        test_pool = self._content_frames("MC", 0, "01", ("d", "e", "f"))
        train_video = build_video_entries(train_pool, namespace="train_pool")[0]
        test_video = build_video_entries(test_pool, namespace="test_pool")[0]
        sidecar = {
            train_video.stable_source_id: {
                "negative_subtype": "near_miss",
                "sample_weight": 0.5,
            }
        }
        applied = apply_sidecar([train_video, test_video], sidecar)
        by_uid = {video.stable_source_id: video for video in applied}
        self.assertEqual(
            by_uid[train_video.stable_source_id].negative_subtype, "near_miss"
        )
        self.assertIsNone(by_uid[test_video.stable_source_id].negative_subtype)
        self.assertEqual(by_uid[test_video.stable_source_id].sample_weight, 1.0)

    def test_collision_classification_uses_configured_identity_mode(self) -> None:
        """Audit B2: the overlap diagnostic must group by the configured mode.

        Under ``game_label_video`` the strict overlap set is keyed
        ``game::label::video``. The classification used to re-derive uids with
        the default ``game_video`` mode, so it grouped frames under ``game::
        video`` and then looked up the label-qualified overlap uid in a mapping
        that never contained it -- a KeyError instead of the friendly
        diagnostic.
        """
        from game_cls.data.indexing import _source_uid_content_classification

        def _one(split: str, content_hash: str) -> FrameRecord:
            return FrameRecord(
                sample_id=f"{split}:MC:0:01:1",
                split=split,
                game="MC",
                label=0,
                video_id="01",
                frame_id=1,
                path="/d/MC/0/01.png",
                width=448,
                height=208,
                channels=3,
                file_size=1,
                content_sha256=content_hash,
            )

        frames_by_split = {
            "train": [_one("train", "hash-a")],
            "test": [_one("test", "hash-a")],
        }
        source_uids_by_split = {"train": {"MC::0::01"}, "test": {"MC::0::01"}}
        classification = _source_uid_content_classification(
            frames_by_split, source_uids_by_split, {}, "game_label_video"
        )
        entries = classification["pairs"].get("train__test", [])
        self.assertEqual(
            [entry["source_video_uid"] for entry in entries], ["MC::0::01"]
        )
        self.assertEqual(entries[0]["shared_content_frames"], 2)

    def test_canonical_uid_roundtrips_through_parquet(self) -> None:
        import tempfile

        from game_cls.data.video_index import (
            build_video_entries,
            read_video_entries_parquet,
            write_video_entries_parquet,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "videos.parquet"
            videos = build_video_entries(
                self._content_frames("MC", 0, "01", ("a", "b", "c"))
            )
            write_video_entries_parquet(videos, path)
            restored = read_video_entries_parquet(path)
            self.assertEqual(len(restored), 1)
            self.assertEqual(
                restored[0].source_version_id, videos[0].source_version_id
            )
            self.assertEqual(
                restored[0].stable_source_id,
                videos[0].stable_source_id,
            )
            self.assertEqual(
                restored[0].content_version_id,
                videos[0].content_version_id,
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
        self.assertEqual(source_video_uid("g", "01", 0, mode="game_video"), "g::01")

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

    def test_uid_accepts_namespace_kwarg(self) -> None:
        # step8: the audit-boundary source-pool namespace prefixes the uid.
        self.assertEqual(
            source_video_uid("g", "01", namespace="train_pool"), "train_pool::g::01"
        )
        self.assertEqual(
            source_video_uid(
                "g", "01", 0, mode="game_label_video", namespace="train_pool"
            ),
            "train_pool::g::0::01",
        )
        # Empty / absent namespace keeps the legacy uid byte-for-byte.
        self.assertEqual(source_video_uid("g", "01", namespace=""), "g::01")
        self.assertEqual(source_video_uid("g", "01", namespace=None), "g::01")


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
        config = load_config("configs/recipes/example_debug.yaml")
        self.assertEqual(config["data"]["source_video_identity"]["mode"], "game_video")

    def test_config_unknown_mode_is_rejected(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        config["data"]["source_video_identity"]["mode"] = "game_label"
        with self.assertRaisesRegex(ConfigSchemaError, "source_video_identity"):
            finalize_config(config)


class SourceIdentityNamespaceConfigTests(unittest.TestCase):
    """step8 source provenance namespaces: forms, resolution and rejection."""

    def test_namespaces_absent_by_default(self) -> None:
        config = load_config("configs/recipes/example_debug.yaml")
        svc = config["data"]["source_video_identity"]
        self.assertNotIn("namespaces", svc)
        self.assertEqual(resolve_source_identity_namespaces(svc), {})

    def test_explicit_namespaces_resolve_to_per_split(self) -> None:
        resolved = resolve_source_identity_namespaces(
            {
                "namespaces": {
                    "train": "train_pool",
                    "val": "train_pool",
                    "test": "heldout_pool",
                }
            }
        )
        self.assertEqual(
            resolved,
            {"train": "train_pool", "val": "train_pool", "test": "heldout_pool"},
        )

    def test_shorthand_namespaces_resolve_train_val_to_source(self) -> None:
        resolved = resolve_source_identity_namespaces(
            {"namespaces": {"source": "train_pool", "test": "heldout_pool"}}
        )
        self.assertEqual(
            resolved,
            {"train": "train_pool", "val": "train_pool", "test": "heldout_pool"},
        )

    def _config_with_namespaces(self, namespaces: dict) -> dict:
        config = load_config("configs/recipes/example_debug.yaml")
        config["data"]["source_video_identity"]["namespaces"] = namespaces
        return config

    def test_train_val_namespace_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "train and val must share"):
            finalize_config(
                self._config_with_namespaces(
                    {"train": "pool_a", "val": "pool_b", "test": "heldout"}
                )
            )

    def test_test_namespace_equals_train_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "must differ"):
            finalize_config(
                self._config_with_namespaces(
                    {"train": "pool", "val": "pool", "test": "pool"}
                )
            )

    def test_shorthand_missing_test_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "requires a test key"):
            finalize_config(self._config_with_namespaces({"source": "train_pool"}))

    def test_mixed_forms_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "mutually exclusive"):
            finalize_config(
                self._config_with_namespaces(
                    {"source": "train_pool", "train": "pool_a", "test": "heldout"}
                )
            )

    def test_empty_namespace_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigSchemaError, "non-empty"):
            finalize_config(
                self._config_with_namespaces(
                    {"train": "pool", "val": "pool", "test": ""}
                )
            )

    def test_aliased_test_namespaces_rejected(self) -> None:
        # When test_index is aliased as validation, declaring distinct test
        # namespaces would bypass the train/val source-identity check.
        config = load_config("configs/recipes/example_debug.yaml")
        data = config["data"]
        data["test_index"] = "indexes/test_frames.parquet"
        data["source_video_identity"]["namespaces"] = {
            "source": "train_pool",
            "test": "heldout_pool",
        }
        with self.assertRaisesRegex(ConfigSchemaError, "aliases test_index"):
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
                write_video_frames(test_root, "game_d", label, "01", [1, 2, 3])
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
            self.assertEqual(manifest["split_source_identity_mode"], "game_label_video")

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
                write_video_frames(test_root, "game_d", label, "01", [1, 2, 3])
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
                write_video_frames(test_root, "game_d", label, "01", [1, 2, 3])
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


@unittest.skipIf(pq is None, "pyarrow is not installed in the current interpreter")
class SourceIdentityNamespaceBundleTests(unittest.TestCase):
    """step8: a distinct test namespace clears coincidental id overlap."""

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
            "source_identity_mode": "game_video",
        }

    def _scanned_config(self) -> tuple[dict, ImageSpec, ScanPolicy, DuplicatePolicy]:
        data_config = self._data_config()
        return (
            data_config,
            ImageSpec.from_config(data_config),
            ScanPolicy.from_config(data_config),
            DuplicatePolicy.from_config(data_config),
        )

    def test_namespaced_bundle_clears_coincidental_test_overlap(self) -> None:
        """train_all and test share video ids 01..03 (physically different
        videos). Distinct namespaces clear the false-positive overlap and the
        strict gate passes under the same namespaces."""
        from game_cls.data.indexing import validate_audit_file

        data_config, image_spec, scan_policy, duplicate_policy = self._scanned_config()
        split_config = self._split_config()
        namespaces_by_split = {
            "train": "train_pool",
            "val": "train_pool",
            "test": "heldout_pool",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_all = root / "train_all"
            for label in (0, 1):
                for video_id in ("01", "02", "03"):
                    write_video_frames(
                        train_all, "game_a", label, video_id, list(range(1, 13))
                    )
            test_root = root / "test"
            # Same game, same video ids as train_all, but every file's bytes
            # are unique (write_unique_png embeds the path), so the content
            # layer sees no identical frame.
            for label in (0, 1):
                for video_id in ("01", "02", "03"):
                    write_video_frames(
                        test_root, "game_a", label, video_id, list(range(1, 5))
                    )
            output = root / "indexes"
            audit = write_split_bundle(
                train_all,
                test_root,
                output,
                image_spec,
                scan_policy,
                duplicate_policy,
                split_config=split_config,
                compute_content_hash=True,
                namespaces_by_split=namespaces_by_split,
            )
            leakage = audit["leakage"]
            self.assertEqual(leakage["source_video_uid_overlap"]["train__test"], [])
            self.assertEqual(leakage["source_video_uid_overlap"]["train__val"], [])
            self.assertEqual(leakage["source_video_uid_overlap"]["val__test"], [])
            self.assertEqual(leakage["video_keys_across_splits"], [])
            self.assertEqual(
                leakage["source_identity_namespaces"],
                dict(sorted(namespaces_by_split.items())),
            )
            validate_audit_file(
                output / "audit.json",
                image_spec=image_spec,
                scan_policy=scan_policy,
                duplicate_policy=duplicate_policy,
                require_test_delta=2,
                namespaces_by_split=namespaces_by_split,
            )

    def test_unconfigured_bundle_has_no_namespace_key(self) -> None:
        data_config, image_spec, scan_policy, duplicate_policy = self._scanned_config()
        split_config = self._split_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_all = root / "train_all"
            for label in (0, 1):
                for video_id in ("01", "02"):
                    write_video_frames(
                        train_all, "game_a", label, video_id, list(range(1, 9))
                    )
            test_root = root / "test"
            for label in (0, 1):
                write_video_frames(test_root, "game_d", label, "01", [1, 2, 3])
            output = root / "indexes"
            audit = write_split_bundle(
                train_all,
                test_root,
                output,
                image_spec,
                scan_policy,
                duplicate_policy,
                split_config=split_config,
                compute_content_hash=True,
            )
            self.assertNotIn("source_identity_namespaces", audit["leakage"])
            self.assertNotIn(
                "source_uid_overlap_content_classification", audit["leakage"]
            )

    def test_identical_content_across_namespaced_splits_still_fatal(self) -> None:
        """The SHA-256 layer is namespace-blind: byte-identical frames crossing
        splits stay fatal even under distinct namespaces."""
        from game_cls.data.indexing import validate_audit_file

        data_config, image_spec, scan_policy, duplicate_policy = self._scanned_config()
        split_config = self._split_config()
        namespaces_by_split = {
            "train": "train_pool",
            "val": "train_pool",
            "test": "heldout_pool",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_all = root / "train_all"
            # Three videos keep every (game, label) stratum splittable.
            for video_id in ("01", "02", "03"):
                write_video_frames(train_all, "game_a", 0, video_id, [1, 2, 3])
            test_root = root / "test"
            # test mirrors video 01 with BYTE-IDENTICAL frames.
            for frame_id in (1, 2, 3):
                name = f"01{frame_id:05d}.png"
                test_path = test_root / "game_a" / "0" / name
                test_path.parent.mkdir(parents=True, exist_ok=True)
                test_path.write_bytes((train_all / "game_a" / "0" / name).read_bytes())
            output = root / "indexes"
            write_split_bundle(
                train_all,
                test_root,
                output,
                image_spec,
                scan_policy,
                duplicate_policy,
                split_config=split_config,
                compute_content_hash=True,
                namespaces_by_split=namespaces_by_split,
            )
            with self.assertRaisesRegex(
                RuntimeError, "identical content crosses split boundaries"
            ):
                validate_audit_file(
                    output / "audit.json",
                    image_spec=image_spec,
                    scan_policy=scan_policy,
                    duplicate_policy=duplicate_policy,
                    require_test_delta=2,
                    namespaces_by_split=namespaces_by_split,
                )


if __name__ == "__main__":
    unittest.main()
