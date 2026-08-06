"""Hard-negative mining manifest (step5 P3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from game_cls.reports.benchmark import (
    read_mining_manifest,
    scan_negative_pool,
    write_mining_manifest,
)


class MiningManifestTests(unittest.TestCase):
    def test_write_read_roundtrip(self) -> None:
        rows = [
            {
                "source_video_uid": "A::01",
                "game": "A",
                "video_id": "01",
                "frame0_id": 1,
                "frame1_id": 3,
                "delta": 2,
                "p_positive": 0.9,
                "subtype_before": "flat_floor",
                "rank_in_video": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard_negatives.parquet"
            write_mining_manifest(rows, path)
            loaded = read_mining_manifest(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["source_video_uid"], "A::01")
            self.assertEqual(loaded[0]["p_positive"], 0.9)
            self.assertEqual(loaded[0]["mining_version"], 1)

    def test_version_mismatch_rejected(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [{"source_video_uid": "A::01", "mining_version": 99}]
                ),
                path,
            )
            with self.assertRaises(ValueError):
                read_mining_manifest(path)


@unittest.skipIf(torch is None, "torch is not installed")
class ScanNegativePoolTests(unittest.TestCase):
    class _ScoringModel(torch.nn.Module):
        def forward(self, image0, image1):
            # p_positive rises with the value planted in image0[...,0,0].
            score = image0[:, 0, 0, 0]
            return torch.stack((-score, score), dim=1)

    def _batch_loader(self):
        images = torch.zeros((6, 2, 3, 8, 8))
        # Per-video planted scores: A::01 [0.9, 0.8], A::02 [0.7, 0.6],
        # B::01 [0.95, 0.95] (duplicate pair identity).
        images[0, 0, 0, 0, 0] = 0.9
        images[1, 0, 0, 0, 0] = 0.8
        images[2, 0, 0, 0, 0] = 0.7
        images[3, 0, 0, 0, 0] = 0.6
        images[4, 0, 0, 0, 0] = 0.95
        images[5, 0, 0, 0, 0] = 0.95
        labels = torch.zeros(6, dtype=torch.long)
        metas = [
            {"game": "A", "video_id": "01", "frame0_id": 1, "frame1_id": 3, "delta": 2, "negative_subtype": "flat_floor"},
            {"game": "A", "video_id": "01", "frame0_id": 2, "frame1_id": 4, "delta": 2, "negative_subtype": "flat_floor"},
            {"game": "A", "video_id": "02", "frame0_id": 1, "frame1_id": 3, "delta": 2, "negative_subtype": "flat_floor"},
            {"game": "A", "video_id": "02", "frame0_id": 2, "frame1_id": 4, "delta": 2, "negative_subtype": "flat_floor"},
            {"game": "B", "video_id": "01", "frame0_id": 1, "frame1_id": 3, "delta": 2, "negative_subtype": None},
            {"game": "B", "video_id": "01", "frame0_id": 1, "frame1_id": 3, "delta": 2, "negative_subtype": None},
        ]
        return [{"images": images, "labels": labels, "meta": metas}]

    def test_topk_per_video_and_dedup(self) -> None:
        model = self._ScoringModel()
        rows = scan_negative_pool(
            model, self._batch_loader(), "cpu", top_k_per_video=1, score_threshold=None, max_samples=None
        )
        by_uid = {row["source_video_uid"]: row["p_positive"] for row in rows}
        # softmax((-score, score))[1] = sigmoid(2*score), so p_positive is
        # monotone in score but not equal to it.
        self.assertEqual(len(rows), 3)
        self.assertTrue(by_uid["A::01"] > by_uid["A::02"])
        self.assertTrue(by_uid["B::01"] > by_uid["A::01"])
        # A::01 keeps its highest-scoring pair (score 0.9 -> sigmoid(1.8)).
        self.assertAlmostEqual(by_uid["A::01"], 0.858, places=3)
        # Rows are ranked best-first.
        self.assertEqual([row["p_positive"] for row in rows], sorted((r["p_positive"] for r in rows), reverse=True))

    def test_score_threshold_filters(self) -> None:
        model = self._ScoringModel()
        rows = scan_negative_pool(
            model, self._batch_loader(), "cpu", top_k_per_video=8, score_threshold=0.75, max_samples=None
        )
        self.assertTrue(all(row["p_positive"] >= 0.75 for row in rows))

    def test_max_samples_caps(self) -> None:
        model = self._ScoringModel()
        rows = scan_negative_pool(
            model, self._batch_loader(), "cpu", top_k_per_video=8, score_threshold=None, max_samples=2
        )
        self.assertEqual(len(rows), 2)

    def test_positives_are_excluded(self) -> None:
        loader = self._batch_loader()
        loader[0]["labels"] = torch.ones(6, dtype=torch.long)
        rows = scan_negative_pool(
            self._ScoringModel(), loader, "cpu", top_k_per_video=8, score_threshold=None, max_samples=None
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
