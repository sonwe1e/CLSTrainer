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
            [1, 2, 3, 4], 10, 2.5, 1.25, torch.device("cpu")
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
                    "meta": {
                        "game": "A",
                        "label": label,
                        "video_id": "01",
                    },
                }

        class Model(torch.nn.Module):
            def forward(self, image0, image1):
                return torch.zeros(len(image0), 2)

        result = evaluate(
            Model(),
            DataLoader(Dataset(), batch_size=2, collate_fn=pair_collate),
            torch.device("cpu"),
        )
        rows = result.grouped_metrics["by_video"]
        self.assertEqual(
            {(row["game"], row["label"], row["video_id"]) for row in rows},
            {("A", 0, "01"), ("A", 1, "01")},
        )
        self.assertIn("brier_score", result.metrics)
        self.assertIn("ece_20_bins", result.metrics)
        self.assertNotIn("macro_video_f1_tau099", result.metrics)


if __name__ == "__main__":
    unittest.main()
