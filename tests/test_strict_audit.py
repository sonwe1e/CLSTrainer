from __future__ import annotations

import unittest
from dataclasses import asdict

from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy
from game_cls.data.indexing import (
    analyze_content_duplicates,
    validate_audit,
)
from game_cls.data.records import FrameRecord

SPEC = ImageSpec(width=448, height=208, channels=3)
POLICY = DuplicatePolicy()


def valid_audit() -> dict:
    split = {
        "frame_count": 4,
        "findings": {
            "errors": [],
            "warnings": [],
            "info": [],
            "ignored": {"counts": {}, "examples": {}},
        },
        "games_missing_labels": {},
        "valid_pairs": {"1": 1, "2": 1, "3": 1},
        "valid_pairs_by_game_label_delta": [
            {
                "game": "game_A",
                "label": label,
                "delta": delta,
                "count": 1,
            }
            for label in (0, 1)
            for delta in (1, 2, 3)
        ],
    }
    return {
        "audit_format_version": 2,
        "expected": {
            "width": SPEC.width,
            "height": SPEC.height,
            "channels": SPEC.channels,
        },
        "policies": {"duplicate_policy": asdict(POLICY)},
        "splits": {"train": dict(split), "test": dict(split)},
        "duplicates": {
            "errors": [],
            "warnings": [],
            "info": [],
            "content_hash_check_enabled": True,
            "same_basename": {
                "group_count": 0,
                "record_count": 0,
                "severity": "info",
            },
        },
        "leakage": {
            "video_keys_across_splits": [],
            "source_identity_mode": "game_video",
        },
    }


def frame(
    *,
    split: str,
    label: int,
    path: str,
    sha256: str,
    frame_id: int = 1,
) -> FrameRecord:
    return FrameRecord(
        sample_id=f"{split}:game_A:{label}:01:{frame_id:05d}",
        split=split,
        game="game_A",
        label=label,
        video_id="01",
        frame_id=frame_id,
        path=path,
        width=448,
        height=208,
        channels=3,
        file_size=1,
        content_sha256=sha256,
    )


class StrictAuditTests(unittest.TestCase):
    def test_accepts_valid_audit_and_scan_warnings(self) -> None:
        audit = valid_audit()
        audit["splits"]["train"]["findings"]["warnings"] = [
            {
                "severity": "warning",
                "kind": "unexpected_nested_directory",
                "path": "/data/game_A/0/random_dir",
            }
        ]
        validate_audit(audit, image_spec=SPEC, duplicate_policy=POLICY)

    def test_rejects_errors_missing_class_and_test_pairs(self) -> None:
        audit = valid_audit()
        audit["splits"]["train"]["findings"]["errors"] = [
            {
                "severity": "error",
                "kind": "unexpected_dimensions",
                "path": "/data/bad.png",
            }
        ]
        audit["splits"]["train"]["games_missing_labels"] = {"game_A": [1]}
        audit["splits"]["test"]["valid_pairs"]["2"] = 0
        with self.assertRaisesRegex(RuntimeError, "Strict dataset audit failed"):
            validate_audit(audit)

    def test_video_key_overlap_is_always_fatal(self) -> None:
        """Cross-split video keys are leakage; no opt-out flag anymore."""
        audit = valid_audit()
        audit["leakage"]["video_keys_across_splits"] = [
            {"game": "game_A", "label": 1, "video_id": "01"}
        ]
        with self.assertRaisesRegex(RuntimeError, "share source video keys"):
            validate_audit(audit, require_content_hash=True)

    def test_source_video_uid_overlap_is_fatal(self) -> None:
        """source_video_uid (game::video_id) must not span splits."""
        audit = valid_audit()
        audit["leakage"]["source_video_uid_overlap"] = {
            "train__test": ["game_A::01"],
            "train__val": [],
            "val__test": [],
        }
        with self.assertRaisesRegex(RuntimeError, "source videos span"):
            validate_audit(audit)
        audit["leakage"]["source_video_uid_overlap"] = {
            "train__test": [],
            "train__val": [],
            "val__test": [],
        }
        validate_audit(audit)

    def test_source_identity_mode_mismatch_is_fatal(self) -> None:
        """An audit built under one identity mode cannot gate another."""
        audit = valid_audit()
        audit["leakage"]["source_identity_mode"] = "game_video"
        with self.assertRaisesRegex(RuntimeError, "identity mode does not match"):
            validate_audit(audit, identity_mode="game_label_video")
        audit["leakage"]["source_identity_mode"] = "game_label_video"
        validate_audit(audit, identity_mode="game_label_video")

    def test_game_label_video_uid_overlap_is_fatal(self) -> None:
        """game::label::video_id uids must not span splits either."""
        from game_cls.data.splitter import source_video_uid

        train = frame(
            split="train", label=0, path="/train/0100001.png", sha256="hash-a"
        )
        test = frame(split="test", label=0, path="/test/0100001.png", sha256="hash-b")
        train_uid = source_video_uid(
            train.game, train.video_id, train.label, mode="game_label_video"
        )
        test_uid = source_video_uid(
            test.game, test.video_id, test.label, mode="game_label_video"
        )
        self.assertEqual(train_uid, "game_A::0::01")
        self.assertEqual(test_uid, "game_A::0::01")
        audit = valid_audit()
        audit["leakage"]["source_identity_mode"] = "game_label_video"
        audit["leakage"]["source_video_uid_overlap"] = {
            "train__test": [train_uid],
            "train__val": [],
            "val__test": [],
        }
        with self.assertRaisesRegex(RuntimeError, "source videos span"):
            validate_audit(audit, identity_mode="game_label_video")

    def test_rejects_insufficient_game_label_delta_coverage(self) -> None:
        audit = valid_audit()
        audit["splits"]["train"]["valid_pairs_by_game_label_delta"][4]["count"] = 0
        with self.assertRaisesRegex(RuntimeError, "label=1/delta=2"):
            validate_audit(
                audit,
                minimum_pairs_per_game_label_delta={2: 1},
            )

    def test_same_label_cross_split_is_fatal(self) -> None:
        """Identical content across splits is leakage, not a warning."""
        duplicates = analyze_content_duplicates(
            {
                "train": [
                    frame(
                        split="train",
                        label=1,
                        path="/train/0100001.png",
                        sha256="same",
                    )
                ],
                "test": [
                    frame(
                        split="test",
                        label=1,
                        path="/test/0100001.png",
                        sha256="same",
                    )
                ],
            },
            POLICY,
        )
        self.assertFalse(duplicates["errors"])
        self.assertEqual(
            duplicates["warnings"][0]["kind"],
            "same_label_content_overlap_across_splits",
        )
        audit = valid_audit()
        audit["duplicates"] = duplicates
        with self.assertRaisesRegex(
            RuntimeError, "identical content crosses split boundaries"
        ):
            validate_audit(audit, require_content_hash=True)

    def test_same_content_with_different_labels_is_fatal(self) -> None:
        duplicates = analyze_content_duplicates(
            {
                "train": [
                    frame(
                        split="train",
                        label=0,
                        path="/train/0100001.png",
                        sha256="same",
                    )
                ],
                "test": [
                    frame(
                        split="test",
                        label=1,
                        path="/test/0100001.png",
                        sha256="same",
                    )
                ],
            },
            POLICY,
        )
        self.assertEqual(
            duplicates["errors"][0]["kind"],
            "identical_content_with_conflicting_labels",
        )
        audit = valid_audit()
        audit["duplicates"] = duplicates
        with self.assertRaisesRegex(RuntimeError, "duplicate conflict"):
            validate_audit(audit, require_content_hash=True)

    def test_same_basename_with_different_content_is_info_only(self) -> None:
        duplicates = analyze_content_duplicates(
            {
                "train": [
                    frame(
                        split="train",
                        label=0,
                        path="/train/0100001.png",
                        sha256="hash-a",
                    )
                ],
                "test": [
                    frame(
                        split="test",
                        label=1,
                        path="/test/0100001.png",
                        sha256="hash-b",
                    )
                ],
            },
            POLICY,
        )
        self.assertFalse(duplicates["errors"])
        self.assertFalse(duplicates["warnings"])
        self.assertEqual(duplicates["same_basename"]["group_count"], 1)
        self.assertEqual(duplicates["same_basename"]["severity"], "info")

    def test_same_label_duplicate_within_split_is_warning(self) -> None:
        duplicates = analyze_content_duplicates(
            {
                "train": [
                    frame(
                        split="train",
                        label=0,
                        path="/train/a/0100001.png",
                        sha256="same",
                    ),
                    frame(
                        split="train",
                        label=0,
                        path="/train/b/0100001.png",
                        sha256="same",
                        frame_id=2,
                    ),
                ]
            },
            POLICY,
        )
        self.assertFalse(duplicates["errors"])
        self.assertEqual(
            duplicates["warnings"][0]["kind"],
            "duplicate_content_within_split",
        )


class NamespaceAuditTests(unittest.TestCase):
    """step8: per-split source namespaces must match the audit and clear
    coincidental train/test overlap."""

    NAMESPACES = {
        "train": "train_pool",
        "val": "train_pool",
        "test": "heldout_pool",
    }

    def test_namespaces_mismatch_is_fatal(self) -> None:
        audit = valid_audit()
        audit["leakage"]["source_identity_namespaces"] = self.NAMESPACES
        audit["leakage"]["source_video_uid_overlap"] = {
            "train__test": [],
            "train__val": [],
            "val__test": [],
        }
        with self.assertRaisesRegex(RuntimeError, "namespaces do not match"):
            validate_audit(audit, namespaces_by_split={"test": "other_pool"})
        validate_audit(audit, namespaces_by_split=self.NAMESPACES)

    def test_old_audit_without_namespaces_requires_rebuild_when_configured(
        self,
    ) -> None:
        # A pre-namespace audit.json has no source_identity_namespaces key; a
        # config that now declares namespaces must force a rebuild.
        audit = valid_audit()
        with self.assertRaisesRegex(RuntimeError, "namespaces do not match"):
            validate_audit(audit, namespaces_by_split=self.NAMESPACES)

    def test_old_audit_without_namespaces_passes_when_unconfigured(self) -> None:
        # Default path unchanged: absent namespaces on both sides validate.
        audit = valid_audit()
        validate_audit(audit)
        validate_audit(audit, namespaces_by_split=None)

    def test_namespaced_stored_overlap_is_honored(self) -> None:
        audit = valid_audit()
        audit["leakage"]["source_identity_namespaces"] = self.NAMESPACES
        # Under namespaces the stored overlap is empty (coincidental ids in
        # distinct pools), so the strict gate passes.
        audit["leakage"]["source_video_uid_overlap"] = {
            "train__test": [],
            "train__val": [],
            "val__test": [],
        }
        audit["leakage"]["video_keys_across_splits"] = []
        validate_audit(audit, namespaces_by_split=self.NAMESPACES)

    def test_content_classification_formatter(self) -> None:
        from game_cls.data.indexing import (
            format_uid_overlap_content_classification,
        )

        text = format_uid_overlap_content_classification(
            {
                "content_hashes_available": True,
                "pairs": {
                    "train__test": [
                        {
                            "source_video_uid": "MC::01",
                            "shared_content_frames": 0,
                        }
                    ]
                },
            }
        )
        self.assertIn("MC::01", text)
        self.assertIn("numbering collision", text)
        self.assertEqual(format_uid_overlap_content_classification({"pairs": {}}), "")


if __name__ == "__main__":
    unittest.main()
