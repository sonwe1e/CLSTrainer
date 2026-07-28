from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
except ImportError:
    np = pa = pq = torch = Image = None


@unittest.skipIf(torch is None, "packed backend dependencies are not installed")
class PackedBackendTests(unittest.TestCase):
    def test_packed_decode_matches_png_pixels(self) -> None:
        from game_cls.data.packed_backend import (
            PackedUint8Backend,
            pack_frame_index,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            expected = {}
            for index in range(3):
                array = np.full((8, 6, 3), index * 70, dtype=np.uint8)
                array[:, :, 1] += 5
                path = root / f"image_{index}.png"
                Image.fromarray(array).save(path)
                rows.append({"path": str(path)})
                expected[str(path)] = torch.from_numpy(
                    array.transpose(2, 0, 1).copy()
                )
            frame_index = root / "frames.parquet"
            pq.write_table(pa.Table.from_pylist(rows), frame_index)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                images_per_shard=2,
                expected_width=6,
                expected_height=8,
            )
            backend = PackedUint8Backend(
                packed_index, channels=3, height=8, width=6
            )
            for path, tensor in expected.items():
                self.assertTrue(torch.equal(backend(path), tensor))
            self.assertEqual(
                len(list((root / "packed").glob("shard_*.bin"))), 2
            )
            backend.close()


if __name__ == "__main__":
    unittest.main()
