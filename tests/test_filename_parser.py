from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from game_cls.data.records import parse_filename, read_png_metadata


class FilenameParserTests(unittest.TestCase):
    def test_valid_filename(self) -> None:
        self.assertEqual(parse_filename("0100001.png"), ("01", 1))
        self.assertEqual(parse_filename("9912345.png"), ("99", 12345))

    def test_invalid_filename(self) -> None:
        for filename in ("100001.png", "0100001.jpg", "AA00001.png", "01000001.png"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                parse_filename(filename)

    def test_reads_png_ihdr_without_pillow(self) -> None:
        header = (
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", 208, 448)
            + bytes([8, 2])
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0100001.png"
            path.write_bytes(header)
            self.assertEqual(read_png_metadata(path), (208, 448, 3))


if __name__ == "__main__":
    unittest.main()
