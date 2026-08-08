from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluationContractTests(unittest.TestCase):
    def test_distributed_reduction_uses_hccl_supported_dtypes(self) -> None:
        from game_cls.engine.evaluator import _make_reduction_tensors

        counts, floating = _make_reduction_tensors(
            [1, 2, 3, 4], 10, 2.5, 1.25, 0.5, torch.device("cpu")
        )
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(floating.dtype, torch.float32)

    def test_near_threshold_boundary_and_high_confidence_count_only(self) -> None:
        from game_cls.engine.evaluator import _threshold_band

        self.assertEqual(_threshold_band(0.990), "0.990-0.995")
        self.assertEqual(_threshold_band(0.995), "0.990-0.995")
        self.assertIsNone(_threshold_band(0.999))

    def test_video_identity_includes_label_and_calibration_metrics_exist(self) -> None:
        from torch.utils.data import DataLoader

        from game_cls.data.collate import pair_collate
        from game_cls.engine.evaluator import evaluate

        class Dataset:
            def __len__(self):
                return 4

            def __getitem__(self, index):
                label = index % 2
                return {
                    "images": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
                    "label": label,
                    "game_id": 0,
                    "game_label_id": label,
                    "video_group_id": label,
                    "meta": {
                        "game": "A",
                        "label": label,
                        "video_id": "01",
                    },
                }

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_dtype = None

            def forward(self, image0, image1):
                self.seen_dtype = image0.dtype
                return torch.zeros(
                    len(image0), 2, dtype=image0.dtype, device=image0.device
                )

        model = Model()
        result = evaluate(
            model,
            DataLoader(Dataset(), batch_size=2, collate_fn=pair_collate),
            torch.device("cpu"),
            amp=True,
            amp_dtype="bfloat16",
            group_catalogs={
                "game": ["A"],
                "game_label": [("A", 0), ("A", 1)],
                "video": [("A", 0, "01"), ("A", 1, "01")],
            },
        )
        fallback = evaluate(
            model,
            DataLoader(Dataset(), batch_size=2, collate_fn=pair_collate),
            torch.device("cpu"),
            amp=True,
            amp_dtype="bfloat16",
        )
        self.assertEqual(model.seen_dtype, torch.bfloat16)
        rows = result.grouped_metrics["by_video"]
        self.assertEqual(
            {(row["game"], row["label"], row["video_id"]) for row in rows},
            {("A", 0, "01"), ("A", 1, "01")},
        )
        self.assertIn("brier_score", result.metrics)
        self.assertIn("ece_20_bins", result.metrics)
        self.assertNotIn("macro_video_f1_tau099", result.metrics)
        by_label = result.grouped_metrics["by_game_label"]
        self.assertTrue(all(row["f1"] is None for row in by_label))
        self.assertEqual(
            {row["primary_metric"] for row in by_label},
            {"positive_recall", "negative_specificity"},
        )
        self.assertEqual(result.grouped_metrics, fallback.grouped_metrics)

    def test_composite_selection_and_worst_game_floor(self) -> None:
        from game_cls.engine.training.selection import (
            _annotate_selection,
            _is_better_model,
        )

        config = {
            "selection_metric": "composite",
            "selection_weights": {
                "global_f1": 0.4,
                "macro_game_f1": 0.4,
                "worst_game_f1": 0.2,
            },
            "minimum_worst_game_f1": 0.5,
        }
        candidate = {
            "global_f1_at_decision_threshold": 0.9,
            "macro_game_f1_at_decision_threshold": 0.8,
            "worst_game_f1_at_decision_threshold": 0.6,
        }
        annotated = _annotate_selection(candidate, config)
        self.assertAlmostEqual(annotated["selection_score"], 0.8)
        self.assertTrue(annotated["selection_eligible"])
        rejected = dict(candidate, worst_game_f1_at_decision_threshold=0.4)
        self.assertFalse(_is_better_model(rejected, {}, config))


if __name__ == "__main__":
    unittest.main()
