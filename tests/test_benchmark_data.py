"""DataLoader throughput probe (step5 P5)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from game_cls.benchmark.data_probe import run_data_probe
from game_cls.data.video_index import VideoEntry, write_video_entries_parquet


@unittest.skipIf(Image is None, "Pillow is not installed")
class BenchmarkDataTests(unittest.TestCase):
    def _dataset(self, directory: Path):
        """Create a tiny real PNG dataset + video-entry parquet."""
        roots = {}
        for label in (0, 1):
            vdir = directory / "game_a" / str(label) / "01"
            vdir.mkdir(parents=True, exist_ok=True)
            frame_ids = []
            for frame_id in (1, 2, 3, 4):
                array = np.random.randint(0, 96, (208, 448, 3), dtype=np.uint8)
                if label:
                    array[:104, :, 0] += 128
                Image.fromarray(array).save(vdir / f"01{frame_id:05d}.png")
                frame_ids.append(frame_id)
            roots[label] = vdir
        entries = [
            VideoEntry(
                game="game_a",
                label=label,
                video_id="01",
                frame_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
                valid_start_positions={2: np.asarray([0, 1], dtype=np.int32)},
                video_directory=str(roots[label]),
                stable_source_id=f"game_a::{label}::01",
                content_version_id=f"content-{label}",
            )
            for label in (0, 1)
        ]
        index_path = directory / "train_video_entries.parquet"
        write_video_entries_parquet(entries, index_path)
        return index_path

    def _config(self, directory: Path) -> dict:
        from game_cls.config import load_config

        index = str(self._dataset(directory))
        config = load_config("configs/recipes/example_debug.yaml")
        config["experiment"]["output_dir"] = str(directory / "runs")
        config["data"].update(
            {
                "synthetic": False,
                "backend": "png",
                "strict_audit": False,
                "train_video_index": index,
                "val_video_index": index,
                "test_video_index": index,
            }
        )
        return config

    def test_probe_produces_well_formed_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            results = run_data_probe(config, steps=2, batch_size=4)
            self.assertTrue(results)
            for metrics in results.values():
                self.assertIn("backend", metrics)
                self.assertIn("samples_per_second", metrics)
                self.assertGreater(metrics["steps"], 0)


if __name__ == "__main__":
    unittest.main()
