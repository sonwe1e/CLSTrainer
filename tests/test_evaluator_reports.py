from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluatorReportTests(unittest.TestCase):
    def test_grouped_metrics_and_report_artifacts(self) -> None:
        from torch.utils.data import DataLoader

        from game_cls.data.collate import pair_collate
        from game_cls.engine.evaluator import evaluate
        from game_cls.engine.trainer import SyntheticPairDataset
        from game_cls.reports.error_writer import (
            write_evaluation_report,
            write_evaluation_shard,
        )

        class RedChannelModel(torch.nn.Module):
            def forward(self, image0, image1):
                score = image0[:, 0].mean(dim=(1, 2)) * 20 - 5
                return torch.stack((-score, score), dim=1)

        dataset = SyntheticPairDataset(16, 32, 16, 77)
        loader = DataLoader(dataset, batch_size=4, collate_fn=pair_collate)
        result = evaluate(
            RedChannelModel(), loader, torch.device("cpu"), checkpoint_step=12
        )
        self.assertEqual(result.metrics["sample_count"], 16)
        self.assertEqual(len(result.grouped_metrics["by_game"]), 2)
        self.assertEqual(len(result.grouped_metrics["by_video"]), 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_evaluation_shard(
                output, 0, result.errors, result.near_threshold
            )
            write_evaluation_report(
                output,
                result.metrics,
                result.grouped_metrics,
                merge_shards=True,
            )
            for name in (
                "metrics.json",
                "metrics_by_game.csv",
                "metrics_by_video.csv",
                "metrics_by_game_label.csv",
                "false_positive.parquet",
                "false_negative.parquet",
                "near_threshold.parquet",
                "errors.html",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

