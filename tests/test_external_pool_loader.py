"""Shared external-pool eval loader (step6 P3).

The benchmark CLI used to build its challenge/mining loaders inline, which
silently dropped three things the training loaders get right: the packed
shard index (so the PNG decoder ran on packed data), the per-pool metadata
sidecar (so ``negative_subtype`` stayed unset and every subtype metric came
back null), and the rank/world_size plumbing. These tests pin the shared
constructor's behaviour on all three, plus the single-process CLI guard.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
except ImportError:  # pragma: no cover - environment guard
    np = pa = pq = torch = Image = None

WIDTH, HEIGHT = 448, 208


def _write_pool(root: Path) -> tuple[Path, dict[int, Path]]:
    """Write a tiny PNG pool: 2 videos (one per label), 4 frames each."""
    rows = []
    directories: dict[int, Path] = {}
    for label in (0, 1):
        video_id = f"0{label + 1}"
        vdir = root / "game_a" / str(label) / video_id
        vdir.mkdir(parents=True, exist_ok=True)
        directories[label] = vdir
        for frame_id in (1, 2, 3, 4):
            array = np.full((HEIGHT, WIDTH, 3), 20 * frame_id + 60 * label, np.uint8)
            path = vdir / f"{video_id}{frame_id:05d}.png"
            Image.fromarray(array).save(path)
            rows.append(
                {
                    "path": str(path),
                    "game": "game_a",
                    "label": label,
                    "video_id": video_id,
                    "frame_id": frame_id,
                }
            )
    frame_index = root / "frames.parquet"
    pq.write_table(pa.Table.from_pylist(rows), frame_index)
    return frame_index, directories


def _video_index(pool: tuple[Path, dict[int, Path]], destination: Path) -> Path:
    from game_cls.data.video_index import VideoEntry, write_video_entries_parquet

    _, directories = pool
    entries = [
        VideoEntry(
            game="game_a",
            label=label,
            video_id=f"0{label + 1}",
            frame_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
            valid_start_positions={2: np.asarray([0, 1], dtype=np.int32)},
            video_directory=str(directories[label]),
        )
        for label in (0, 1)
    ]
    write_video_entries_parquet(entries, destination)
    return destination


def _sidecar(root: Path, name: str = "pool_metadata.parquet") -> Path:
    from game_cls.data.sidecar import write_metadata_sidecar

    path = root / name
    write_metadata_sidecar(
        [{"source_video_uid": "game_a::01", "negative_subtype": "bridge"}], path
    )
    return path


def _base_config(output_dir: Path) -> dict:
    from game_cls.config import load_config

    config = load_config("configs/recipes/example_debug.yaml")
    config["experiment"]["output_dir"] = str(output_dir / "runs")
    config["data"].update({"synthetic": False, "strict_audit": False})
    config["data"]["width"] = WIDTH
    config["data"]["height"] = HEIGHT
    config["train"]["local_batch_size"] = 2
    config["evaluation"]["group_by_negative_subtype"] = True
    config["data"]["mining"]["pool_metadata"] = str(_sidecar(output_dir))
    return config


@unittest.skipIf(torch is None, "external pool loader dependencies missing")
class ExternalPoolLoaderTests(unittest.TestCase):
    def test_packed_pool_uses_the_packed_index_and_sidecar(self) -> None:
        from game_cls.data.image_spec import ImageSpec
        from game_cls.data.packed_backend import PackedUint8Backend, pack_frame_index
        from game_cls.engine.training.loaders import build_external_pool_loader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_index, _ = _write_pool(root)
            packed_index = pack_frame_index(
                frame_index,
                root / "packed",
                image_spec=ImageSpec(width=WIDTH, height=HEIGHT, channels=3),
                images_per_shard=3,
            )
            config = _base_config(root)
            config["data"]["backend"] = "packed_uint8"
            config["data"]["mining"].update(
                {
                    "pool_packed_video_index": str(
                        root / "packed" / "packed_video_entries.parquet"
                    ),
                    "pool_packed_index": str(packed_index),
                }
            )

            loader, videos = build_external_pool_loader(config, pool="mining")

            # The packed decoder is wired in, not the PNG fallback.
            self.assertIsInstance(loader.dataset.decoder, PackedUint8Backend)
            # The sidecar was joined, so subtype grouping is live.
            by_uid = {video.source_video_uid: video for video in videos}
            self.assertEqual(by_uid["game_a::01"].negative_subtype, "bridge")
            self.assertIn("game_label_subtype", loader.dataset.group_catalogs)
            # And the pairs actually decode at the configured shape.
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["images"].shape[1:]), (2, 3, HEIGHT, WIDTH))
            # Windows keeps the memmapped shards locked until they close.
            loader.dataset.decoder.close()

    def test_packed_backend_without_shard_index_is_rejected(self) -> None:
        from game_cls.engine.training.loaders import build_external_pool_loader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _video_index(_write_pool(root), root / "pool_videos.parquet")
            config = _base_config(root)
            config["data"]["backend"] = "packed_uint8"
            config["data"]["mining"]["pool_video_index"] = str(index)

            # Falling back to the PNG decoder on packed shards would produce
            # garbage pixels, so this must fail loudly instead.
            with self.assertRaisesRegex(ValueError, "pool_packed_index"):
                build_external_pool_loader(config, pool="mining")

    def test_missing_video_index_is_rejected(self) -> None:
        from game_cls.engine.training.loaders import build_external_pool_loader

        with tempfile.TemporaryDirectory() as directory:
            config = _base_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "challenge_video_index"):
                build_external_pool_loader(config, pool="challenge")
            with self.assertRaisesRegex(ValueError, "pool_video_index"):
                build_external_pool_loader(config, pool="mining")

    def test_unknown_pool_is_rejected(self) -> None:
        from game_cls.engine.training.loaders import build_external_pool_loader

        with tempfile.TemporaryDirectory() as directory:
            config = _base_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "Unsupported external pool"):
                build_external_pool_loader(config, pool="val")

    def test_challenge_pool_reads_its_own_keys(self) -> None:
        from game_cls.engine.training.loaders import build_external_pool_loader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _video_index(_write_pool(root), root / "challenge_videos.parquet")
            config = _base_config(root)
            config["data"]["backend"] = "png"
            config["data"]["challenge_video_index"] = str(index)

            loader, videos = build_external_pool_loader(config, pool="challenge")

            self.assertEqual(len(videos), 2)
            self.assertGreater(len(loader.dataset), 0)
            # png backend keeps the default (PNG) decoder, not a packed one.
            from game_cls.data.packed_backend import PackedUint8Backend

            self.assertNotIsInstance(loader.dataset.decoder, PackedUint8Backend)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["images"].shape[1:]), (2, 3, HEIGHT, WIDTH))

    def test_pool_shards_across_ranks(self) -> None:
        from game_cls.engine.training.loaders import build_external_pool_loader

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = _video_index(_write_pool(root), root / "pool_videos.parquet")
            config = _base_config(root)
            config["data"]["mining"]["pool_video_index"] = str(index)

            whole, _ = build_external_pool_loader(config, pool="mining")
            shards = [
                build_external_pool_loader(
                    config, pool="mining", rank=rank, world_size=2
                )[0]
                for rank in (0, 1)
            ]

            # Sharding splits the pairs; it must not duplicate or drop them.
            self.assertEqual(
                sum(len(loader.dataset) for loader in shards),
                len(whole.dataset),
            )


@unittest.skipIf(torch is None, "external pool loader dependencies missing")
class SingleProcessGuardTests(unittest.TestCase):
    def test_guard_passes_single_process(self) -> None:
        import os
        from unittest import mock

        from game_cls.cli.benchmark import _reject_multi_process

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_reject_multi_process("challenge"))
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
            self.assertIsNone(_reject_multi_process("challenge"))

    def test_guard_rejects_torchrun(self) -> None:
        import os
        from unittest import mock

        from game_cls.cli.benchmark import _reject_multi_process

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "3"}):
            message = _reject_multi_process("scan-negatives")
        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("scan-negatives", message)
        self.assertIn("WORLD_SIZE=8", message)
        self.assertIn("RANK=3", message)


if __name__ == "__main__":
    unittest.main()
