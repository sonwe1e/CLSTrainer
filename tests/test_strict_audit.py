from __future__ import annotations

import unittest

from game_cls.data.indexing import validate_audit


def valid_audit() -> dict:
    split = {
        "frame_count": 4,
        "unexpected_dimension_count": 0,
        "parse_or_file_issues": [],
        "games_missing_labels": {},
        "valid_pairs": {"1": 1, "2": 1, "3": 1},
        "valid_pairs_by_game_label_delta": [
            {"game": "game_A", "label": label, "delta": delta, "count": 1}
            for label in (0, 1)
            for delta in (1, 2, 3)
        ],
    }
    return {"splits": {"train": dict(split), "test": dict(split)}}


class StrictAuditTests(unittest.TestCase):
    def test_accepts_valid_audit(self) -> None:
        validate_audit(valid_audit())

    def test_rejects_bad_dimensions_missing_class_and_test_pairs(self) -> None:
        audit = valid_audit()
        audit["splits"]["train"]["unexpected_dimension_count"] = 1
        audit["splits"]["train"]["games_missing_labels"] = {"game_A": [1]}
        audit["splits"]["test"]["valid_pairs"]["2"] = 0
        with self.assertRaisesRegex(RuntimeError, "Strict dataset audit failed"):
            validate_audit(audit)

    def test_video_key_overlap_is_optional_but_hash_leakage_is_fatal(self) -> None:
        audit = valid_audit()
        audit["leakage"] = {
            "video_keys_across_splits": [
                {"game": "game_A", "label": 1, "video_id": "01"}
            ],
            "content_hashes_across_splits": [],
            "content_hash_check_enabled": True,
        }
        validate_audit(audit, require_content_hash=True)
        with self.assertRaisesRegex(RuntimeError, "share video keys"):
            validate_audit(
                audit,
                require_content_hash=True,
                require_unique_video_keys=True,
            )
        audit["leakage"]["content_hashes_across_splits"] = [
            {"sha256": "abc"}
        ]
        with self.assertRaisesRegex(RuntimeError, "identical file hashes"):
            validate_audit(audit, require_content_hash=True)

    def test_rejects_insufficient_game_label_delta_coverage(self) -> None:
        audit = valid_audit()
        audit["splits"]["train"][
            "valid_pairs_by_game_label_delta"
        ][4]["count"] = 0
        with self.assertRaisesRegex(RuntimeError, "label=1/delta=2"):
            validate_audit(
                audit,
                minimum_pairs_per_game_label_delta={2: 1},
            )


if __name__ == "__main__":
    unittest.main()
