"""CLI-level tests for the from_train dataset prepare/audit/pack pipeline.

These drive the real subcommand handlers (``cmd_dataset_prepare`` etc.)
and the ``main()`` dispatch fix, so they cover the exact argv wiring users
hit with ``cls-trainer dataset prepare ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from game_cls.cli import (
    build_parser,
    cmd_dataset_audit,
    cmd_dataset_pack,
    cmd_dataset_prepare,
    main,
)
from game_cls.data.indexing import read_frame_parquet
from game_cls.data.splitter import load_split_manifest


def write_png(path: Path, width: int = 448, height: int = 208) -> None:
    """Write a real, decodable PNG whose bytes are unique per file.

    The seed is derived from the resolved path so the audit's content-hash
    duplicate check sees distinct content, and PIL can decode the file for
    the pack backend.
    """
    import numpy as np
    from PIL import Image

    seed = int.from_bytes(
        hashlib.sha256(str(path.resolve()).encode()).digest()[:8], "big"
    )
    array = np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def write_video(
    root: Path, game: str, label: int, video_id: str, frame_count: int
) -> None:
    for frame_id in range(1, frame_count + 1):
        write_png(root / game / str(label) / f"{video_id}{frame_id:05d}.png")


def _write_config(directory: Path) -> Path:
    config = directory / "split_config.yaml"
    config.write_text(
        """\
data:
  width: 448
  height: 208
  channels: 3
  frame_extensions: [".png"]
  ignore_directory_prefixes: ["_", "."]
  ignore_directory_names: ["__pycache__", "cache", "caches", "tmp", "temp"]
  ignore_file_globs: ["*.tmp", "*.part", "*.log"]
  unexpected_nested_directory_severity: warning
  duplicate_policy:
    same_label_cross_split: warning
    same_label_within_split: warning
    cross_label_same_content: warning
    same_basename: info
  split:
    mode: from_train
    val_ratio: 0.2
    seed: 20260728
    target_delta: 2
    manifest: split_manifest.parquet
    on_new_groups: error
    small_stratum_policy: error
pair:
  test_delta: 2
""",
        encoding="utf-8",
    )
    return config


def _build_roots(base: Path) -> None:
    # video ids must be unique per game (source_video_uid is
    # label-independent), so a per-game counter spans both labels.
    train_all = base / "train_all"
    for game in ("game_a", "game_b"):
        for video in range(4):
            label = video % 2
            write_video(train_all, game, label, f"{video + 1:02d}", 4)
    test_root = base / "test"
    for label in (0, 1):
        write_video(test_root, "game_c", label, f"{label + 1:02d}", 3)


def _prepare_dataset(base: Path) -> tuple[Path, Path]:
    _build_roots(base)
    config = _write_config(base)
    output_dir = base / "indexes"
    args = argparse.Namespace(
        config=str(config),
        train_root=str(base / "train_all"),
        test_root=str(base / "test"),
        output_dir=str(output_dir),
        val_ratio=None,
        overrides=[],
    )
    assert cmd_dataset_prepare(args) == 0
    return config, output_dir


class DatasetCliTests(unittest.TestCase):
    def test_dataset_prepare_writes_split_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, output_dir = _prepare_dataset(base)

            self.assertTrue((output_dir / "split_manifest.parquet").is_file())
            self.assertTrue((output_dir / "split_summary.json").is_file())
            self.assertTrue((output_dir / "audit.json").is_file())
            for split in ("train", "val", "test"):
                self.assertTrue((output_dir / f"{split}_frames.parquet").is_file())
                self.assertTrue((output_dir / f"{split}_videos.parquet").is_file())
                self.assertTrue(
                    (output_dir / f"{split}_video_entries.parquet").is_file()
                )

            train_frames = read_frame_parquet(output_dir / "train_frames.parquet")
            val_frames = read_frame_parquet(output_dir / "val_frames.parquet")
            self.assertTrue(train_frames)
            self.assertTrue(val_frames)
            self.assertTrue(all(frame.split == "train" for frame in train_frames))
            self.assertTrue(all(frame.split == "val" for frame in val_frames))

            manifest = load_split_manifest(output_dir / "split_manifest.parquet")
            self.assertEqual(manifest["split_seed"], 20260728)
            self.assertEqual(len(manifest["assignment"]), 8)

    def test_dataset_audit_strict_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config, output_dir = _prepare_dataset(base)
            args = argparse.Namespace(
                config=str(config),
                index_dir=str(output_dir),
                strict=True,
                overrides=[],
            )
            self.assertEqual(cmd_dataset_audit(args), 0)

    def test_dataset_pack_packs_train_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config, output_dir = _prepare_dataset(base)
            packed = base / "packed"
            args = argparse.Namespace(
                config=str(config),
                frame_index=str(output_dir / "train_frames.parquet"),
                output_dir=str(packed),
                images_per_shard=4096,
                overrides=[],
            )
            self.assertEqual(cmd_dataset_pack(args), 0)
            self.assertTrue((packed / "packed_frames.parquet").is_file())
            self.assertTrue((packed / "packed_manifest.json").is_file())
            self.assertTrue((packed / "packed_video_entries.parquet").is_file())
            manifest = json.loads(
                (packed / "packed_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["width"], 448)
            self.assertEqual(manifest["height"], 208)
            self.assertEqual(manifest["channels"], 3)
            packed_count = len(read_frame_parquet(output_dir / "train_frames.parquet"))
            self.assertEqual(manifest["frame_count"], packed_count)

    def test_main_dispatch_recognizes_evaluate_and_dataset(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--run", "latest"])
        self.assertEqual(args.command, "evaluate")
        args = parser.parse_args(
            [
                "dataset",
                "prepare",
                "--config",
                "cfg.yaml",
                "--train-root",
                "a",
                "--test-root",
                "b",
            ]
        )
        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.dataset_command, "prepare")

    def test_main_runs_dataset_prepare_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_roots(base)
            config = _write_config(base)
            output_dir = base / "main_indexes"
            code = main(
                [
                    "dataset",
                    "prepare",
                    "--config",
                    str(config),
                    "--train-root",
                    str(base / "train_all"),
                    "--test-root",
                    str(base / "test"),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "split_manifest.parquet").is_file())
            self.assertTrue((output_dir / "audit.json").is_file())
            self.assertTrue((output_dir / "split_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
