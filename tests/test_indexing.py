from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from game_cls.data.image_spec import ImageSpec
from game_cls.data.index_policy import DuplicatePolicy, ScanPolicy
from game_cls.data.indexing import scan_split, write_index_bundle


def write_png_header(
    path: Path, width: int = 448, height: int = 208
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + bytes([8, 2])
    )


def scan_policy(**overrides) -> ScanPolicy:
    config = {
        "frame_extensions": [".png"],
        "ignore_directory_prefixes": ["_", "."],
        "ignore_directory_names": [
            "__pycache__",
            "cache",
            "caches",
            "tmp",
            "temp",
        ],
        "ignore_file_globs": ["*.tmp", "*.part", "*.log"],
        "unexpected_nested_directory_severity": "warning",
        **overrides,
    }
    return ScanPolicy.from_config(config)


class ImageSpecAndScanTests(unittest.TestCase):
    def test_validates_training_pair_batch_shape(self) -> None:
        import torch

        spec = ImageSpec(width=448, height=208, channels=3)
        images = torch.zeros((2, 2, 3, 208, 448))
        spec.validate_pair_batch_shape(images.shape)
        with self.assertRaisesRegex(
            RuntimeError, "Training image shape mismatch"
        ):
            spec.validate_pair_batch_shape((2, 2, 3, 448, 208))

    def test_uses_configured_landscape_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png_header(root / "game_A" / "0" / "0100001.png")
            write_png_header(
                root / "game_A" / "0" / "0100002.png",
                width=208,
                height=448,
            )
            result = scan_split(
                root,
                "train",
                ImageSpec(width=448, height=208, channels=3),
                scan_policy=scan_policy(),
                compute_content_hash=False,
            )
            self.assertEqual(len(result.frames), 1)
            self.assertEqual(
                (result.frames[0].width, result.frames[0].height),
                (448, 208),
            )
            self.assertEqual(
                result.findings.errors[0]["kind"],
                "unexpected_dimensions",
            )

    def test_different_configured_dimensions_require_no_code_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png_header(
                root / "game_A" / "0" / "0100001.png",
                width=320,
                height=120,
            )
            result = scan_split(
                root,
                "train",
                ImageSpec(width=320, height=120, channels=3),
                scan_policy=scan_policy(),
                compute_content_hash=False,
            )
            self.assertEqual(len(result.frames), 1)
            self.assertFalse(result.findings.errors)

    def test_structured_scan_ignores_auxiliary_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png_header(root / "game_A" / "0" / "0100001.png")
            (root / "game_A" / "0" / "video01.mp4").write_bytes(b"video")
            (root / "game_A" / "0" / "meta.json").write_text("{}")
            for index in range(25):
                (
                    root / "game_A" / "0" / f"aux_{index:02d}.json"
                ).write_text("{}")
            write_png_header(
                root / "game_A" / "0" / "_cache" / "thumb.png"
            )
            write_png_header(root / "game_A" / "_cache" / "foo.png")
            write_png_header(root / "game_A" / "0" / "bad_name.png")
            write_png_header(root / "game_A" / "2" / "0100001.png")
            write_png_header(
                root / "game_A" / "0" / "random_dir" / "file.png"
            )

            result = scan_split(
                root,
                "train",
                ImageSpec(width=448, height=208, channels=3),
                scan_policy=scan_policy(),
                compute_content_hash=False,
            )

            self.assertEqual(
                [Path(frame.path).name for frame in result.frames],
                ["0100001.png"],
            )
            self.assertEqual(
                {item["kind"] for item in result.findings.errors},
                {"invalid_filename", "invalid_label_directory"},
            )
            self.assertEqual(
                [item["kind"] for item in result.findings.warnings],
                ["unexpected_nested_directory"],
            )
            self.assertEqual(
                result.findings.ignored_counts["non_frame_extension"], 27
            )
            self.assertEqual(
                len(
                    result.findings.ignored_examples[
                        "non_frame_extension"
                    ]
                ),
                20,
            )
            self.assertEqual(
                result.findings.ignored_counts["ignored_directory"], 2
            )

            strict_nested = scan_split(
                root,
                "train",
                ImageSpec(width=448, height=208, channels=3),
                scan_policy=scan_policy(
                    unexpected_nested_directory_severity="error"
                ),
                compute_content_hash=False,
            )
            self.assertIn(
                "unexpected_nested_directory",
                {item["kind"] for item in strict_nested.findings.errors},
            )


@unittest.skipIf(
    pq is None, "pyarrow is not installed in the current interpreter"
)
class IndexBundleTests(unittest.TestCase):
    def test_writes_frame_video_indexes_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                for frame_id in (1, 2, 4):
                    write_png_header(
                        root
                        / split
                        / "game_A"
                        / "0"
                        / f"01{frame_id:05d}.png"
                    )
            output = root / "indexes"
            audit = write_index_bundle(
                root / "train",
                root / "test",
                output,
                ImageSpec(width=448, height=208, channels=3),
                scan_policy(),
                DuplicatePolicy(),
            )
            self.assertEqual(audit["audit_format_version"], 3)
            self.assertEqual(
                audit["expected"],
                {"width": 448, "height": 208, "channels": 3},
            )
            self.assertEqual(
                audit["splits"]["train"]["valid_pairs"]["2"], 1
            )
            self.assertEqual(
                audit["splits"]["train"][
                    "valid_pairs_by_game_label_delta"
                ][1],
                {
                    "game": "game_A",
                    "label": 0,
                    "delta": 2,
                    "count": 1,
                },
            )
            self.assertEqual(
                pq.read_table(output / "train_frames.parquet").num_rows, 3
            )
            videos = pq.read_table(
                output / "train_videos.parquet"
            ).to_pylist()
            self.assertEqual(videos[0]["valid_pair_count_delta2"], 1)
            self.assertTrue((output / "audit.json").is_file())
            self.assertTrue(
                (output / "train_video_entries.parquet").is_file()
            )
            video_entries = pq.read_table(
                output / "train_video_entries.parquet"
            ).to_pylist()
            self.assertEqual(video_entries[0]["frame_ids"], [1, 2, 4])
            self.assertEqual(
                video_entries[0]["valid_starts_delta2"], [1]
            )
            self.assertIsNone(video_entries[0]["frame_paths"])
            self.assertTrue(video_entries[0]["video_directory"])
            self.assertTrue(audit["duplicates"]["warnings"])

    def test_three_split_bundle_and_source_uid_leakage(self) -> None:
        """A source video spanning train/val/test must be detected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Same video id '01' in train, val and test: leakage.
            for split in ("train", "val", "test"):
                write_png_header(
                    root / split / "game_A" / "0" / "0100001.png"
                )
            output = root / "indexes"
            audit = write_index_bundle(
                root / "train",
                root / "test",
                output,
                ImageSpec(width=448, height=208, channels=3),
                scan_policy(),
                DuplicatePolicy(),
                val_root=root / "val",
            )
            self.assertEqual(
                audit["leakage"]["source_video_uid_overlap"][
                    "train__test"
                ],
                ["game_A::01"],
            )
            self.assertEqual(
                audit["leakage"]["source_video_uid_overlap"][
                    "train__val"
                ],
                ["game_A::01"],
            )
            with self.assertRaisesRegex(
                RuntimeError, "source videos span"
            ):
                from game_cls.data.indexing import validate_audit

                validate_audit(
                    audit,
                    image_spec=ImageSpec(width=448, height=208, channels=3),
                    duplicate_policy=DuplicatePolicy(),
                    require_content_hash=False,
                )


if __name__ == "__main__":
    unittest.main()
