from __future__ import annotations

import struct
import tempfile
from pathlib import Path
import unittest

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from game_cls.data.indexing import write_index_bundle


def write_png_header(path: Path, width: int = 208, height: int = 448) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + bytes([8, 2])
    )


@unittest.skipIf(pq is None, "pyarrow is not installed in the current interpreter")
class IndexBundleTests(unittest.TestCase):
    def test_writes_frame_video_indexes_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                for frame_id in (1, 2, 4):
                    write_png_header(
                        root / split / "game_A" / "0" / f"01{frame_id:05d}.png"
                    )
            output = root / "indexes"
            audit = write_index_bundle(root / "train", root / "test", output)
            self.assertEqual(audit["splits"]["train"]["valid_pairs"]["2"], 1)
            self.assertEqual(
                pq.read_table(output / "train_frames.parquet").num_rows, 3
            )
            videos = pq.read_table(output / "train_videos.parquet").to_pylist()
            self.assertEqual(videos[0]["valid_pair_count_delta2"], 1)
            self.assertTrue((output / "audit.json").is_file())
            self.assertTrue(
                (output / "train_video_entries.parquet").is_file()
            )
            video_entries = pq.read_table(
                output / "train_video_entries.parquet"
            ).to_pylist()
            self.assertEqual(video_entries[0]["frame_ids"], [1, 2, 4])
            self.assertEqual(video_entries[0]["valid_starts_delta2"], [1])


if __name__ == "__main__":
    unittest.main()
