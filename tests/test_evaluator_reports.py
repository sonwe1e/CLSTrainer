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
        from game_cls.data.image_spec import ImageSpec
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

        dataset = SyntheticPairDataset(
            16, ImageSpec(width=16, height=32, channels=3), 77
        )
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

    def test_packed_preview_export_and_buffered_row_groups(self) -> None:
        import pyarrow.parquet as pq

        from game_cls.reports.error_writer import (
            EvaluationShardWriter,
            write_evaluation_report,
        )

        def row(frame_index: int) -> dict:
            return {
                "game": "A",
                "label": 1,
                "video_id": "01",
                "frame0_id": frame_index,
                "frame1_id": frame_index + 2,
                "delta": 2,
                "image0_path": f"packed://frame/{frame_index}",
                "image1_path": f"packed://frame/{frame_index + 2}",
                "logit0": 1.0,
                "logit1": 0.0,
                "margin": -1.0,
                "probability_class1": 0.25,
                "prediction": 0,
                "error_type": "FN",
                "checkpoint_step": 1,
            }

        def decoder(frame_index: int):
            return torch.full(
                (3, 8, 6), frame_index % 255, dtype=torch.uint8
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with EvaluationShardWriter(
                output, 0, row_group_size=2
            ) as writer:
                writer.write([row(123)], [])
                writer.write([row(125)], [])
                writer.write([row(127)], [])
            shard = (
                output / "shards" / "errors_rank_0000.parquet"
            )
            self.assertEqual(pq.ParquetFile(shard).num_row_groups, 2)
            write_evaluation_report(
                output,
                {"sample_count": 3},
                merge_shards=True,
                html_max_errors=2,
                preview_decoder=decoder,
            )
            preview = output / "previews" / "frame_000000123.png"
            self.assertTrue(preview.is_file())
            html = (output / "errors.html").read_text(encoding="utf-8")
            self.assertIn("previews/frame_000000123.png", html)
            self.assertNotIn("packed://frame/123", html)


if __name__ == "__main__":
    unittest.main()
