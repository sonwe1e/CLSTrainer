"""Negative-subtype grouping and worst-subtype metrics (step5 P2)."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from game_cls.data.lazy_pair_dataset import EvalPairDataset
from game_cls.data.video_index import build_video_entries
from game_cls.engine.evaluator import (
    _build_subtype_group,
    _subtype_negative_counts,
    _worst_subtype_fpr_and_recall,
)
from tests.test_video_index_sampler import frame


def _videos_with_subtypes():
    entries = build_video_entries(
        [frame("A", 0, "01", item) for item in (1, 2, 3, 4)]
        + [frame("A", 0, "02", item) for item in (1, 2, 3)]
    )
    return [
        replace(entries[0], negative_subtype="wooden_bridge"),
        replace(entries[1], negative_subtype="flat_floor"),
    ]


class SubtypeMetricHelpersTests(unittest.TestCase):
    def test_worst_subtype_fpr_and_recall(self) -> None:
        counters = {
            ("A", 0, "wooden_bridge"): [10, 2, 1, 88],
            ("A", 0, "flat_floor"): [5, 0, 0, 50],
        }
        worst_fpr, worst_recall = _worst_subtype_fpr_and_recall(counters)
        self.assertAlmostEqual(worst_fpr, 2 / 90)
        self.assertAlmostEqual(worst_recall, 10 / 11)

    def test_empty_subtypes_return_none(self) -> None:
        self.assertEqual(_worst_subtype_fpr_and_recall({}), (None, None))

    def test_subtype_negative_counts(self) -> None:
        counters = {("A", 0, "x"): [1, 2, 3, 4], ("A", 0, "y"): [0, 5, 0, 5]}
        self.assertEqual(
            _subtype_negative_counts(counters),
            {("A", 0, "x"): 6, ("A", 0, "y"): 10},
        )

    def test_build_subtype_group_skips_zero_rows(self) -> None:
        catalogs = {
            "game": ["A"],
            "game_label": [("A", 0)],
            "video": [("A", 0, "01")],
            "game_label_subtype": [
                ("A", 0, "wooden_bridge"),
                ("A", 0, "flat_floor"),
            ],
        }
        array = np.array([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=np.int64)
        group = _build_subtype_group(catalogs, array)
        self.assertEqual(group, {("A", 0, "flat_floor"): [1, 2, 3, 4]})

    def test_build_subtype_group_absent_catalog(self) -> None:
        self.assertEqual(_build_subtype_group({"game": []}, None), {})


@unittest.skipIf(torch is None, "torch is not installed")
class EvalDatasetSubtypeCatalogTests(unittest.TestCase):
    def _stub_decoder(self, reference):
        return torch.zeros((2, 3, 8, 8))

    def test_group_catalog_and_batch_ids(self) -> None:
        videos = _videos_with_subtypes()
        dataset = EvalPairDataset(
            videos,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([2, 2], dtype=np.int8),
            np.asarray([0, 0], dtype=np.int32),
            decoder=self._stub_decoder,
            group_by_negative_subtype=True,
        )
        self.assertIn("game_label_subtype", dataset.group_catalogs)
        sample0 = dataset[0]
        sample1 = dataset[1]
        self.assertIn("game_label_subtype_id", sample0)
        # The two videos carry distinct subtypes -> distinct ids.
        self.assertNotEqual(
            sample0["game_label_subtype_id"], sample1["game_label_subtype_id"]
        )

    def test_group_catalog_absent_when_disabled(self) -> None:
        videos = _videos_with_subtypes()
        dataset = EvalPairDataset(
            videos,
            np.asarray([0], dtype=np.int32),
            np.asarray([2], dtype=np.int8),
            np.asarray([0], dtype=np.int32),
            decoder=self._stub_decoder,
            group_by_negative_subtype=False,
        )
        self.assertNotIn("game_label_subtype", dataset.group_catalogs)


if __name__ == "__main__":
    unittest.main()
