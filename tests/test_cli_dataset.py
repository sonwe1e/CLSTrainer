"""CLI-level tests for the from_train dataset prepare/audit/pack pipeline.

These drive the real subcommand handlers (``cmd_dataset_prepare`` etc.)
and the ``main()`` dispatch fix, so they cover the exact argv wiring users
hit with ``cls-trainer dataset prepare ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from game_cls.cli import (
    build_parser,
    cmd_dataset_audit,
    cmd_dataset_pack,
    cmd_dataset_prepare,
    main,
)
from game_cls.cli.dataset import (
    _maybe_prepare_split,
    _mining_rows,
    _missing_split_artifacts,
    cmd_dataset_annotate,
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


def _prepare_if_missing_config(base: Path) -> dict:
    """Minimal config that turns ``data.prepare_if_missing`` on."""
    indexes = base / "indexes"
    return {
        "data": {
            "prepare_if_missing": True,
            "source_root": str(base / "train_all"),
            "test_root": str(base / "test"),
            "train_index": str(indexes / "train_frames.parquet"),
            "val_index": str(indexes / "val_frames.parquet"),
            "test_index": str(indexes / "test_frames.parquet"),
            "train_video_index": str(indexes / "train_video_entries.parquet"),
            "val_video_index": str(indexes / "val_video_entries.parquet"),
            "test_video_index": str(indexes / "test_video_entries.parquet"),
            "audit_path": str(indexes / "audit.json"),
            "split": {"mode": "from_train", "val_ratio": 0.2, "seed": 1},
        }
    }


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

    def _build_overlap_roots(self, base: Path) -> None:
        """train_all and test reuse the same (game, video_id) numbering but
        are physically different videos (each PNG's bytes are unique)."""
        train_all = base / "train_all"
        for game in ("game_a", "game_b"):
            for video in range(4):
                label = video % 2
                write_video(train_all, game, label, f"{video + 1:02d}", 4)
        test_root = base / "test"
        for label in (0, 1):
            write_video(test_root, "game_a", label, f"{label + 1:02d}", 3)

    def _write_namespaced_config(self, directory: Path) -> Path:
        config = _write_config(directory)
        text = config.read_text(encoding="utf-8")
        text = text.replace(
            "  duplicate_policy:",
            "  source_video_identity:\n"
            "    namespaces:\n"
            "      source: train_pool\n"
            "      test: heldout_pool\n"
            "  duplicate_policy:",
        )
        config.write_text(text, encoding="utf-8")
        return config

    def test_namespaced_coincidental_ids_pass_strict_audit(self) -> None:
        """step8 end-to-end: declaring distinct source pools clears the
        false-positive source-video overlap and the strict gate passes."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self._build_overlap_roots(base)
            config = self._write_namespaced_config(base)
            output_dir = base / "indexes"
            prepare_args = argparse.Namespace(
                config=str(config),
                train_root=str(base / "train_all"),
                test_root=str(base / "test"),
                output_dir=str(output_dir),
                val_ratio=None,
                overrides=[],
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(cmd_dataset_prepare(prepare_args), 0)
            self.assertIn("source identity namespaces", buffer.getvalue())
            audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(
                audit["leakage"]["source_identity_namespaces"],
                {"train": "train_pool", "val": "train_pool", "test": "heldout_pool"},
            )
            audit_args = argparse.Namespace(
                config=str(config),
                index_dir=str(output_dir),
                strict=True,
                overrides=[],
            )
            self.assertEqual(cmd_dataset_audit(audit_args), 0)

    def test_unconfigured_coincidental_ids_fail_with_classification(self) -> None:
        """Without namespaces the same data fails the strict gate, but the
        content-collision classification is printed first (diagnostic)."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self._build_overlap_roots(base)
            config = _write_config(base)  # no namespaces
            output_dir = base / "indexes"
            prepare_args = argparse.Namespace(
                config=str(config),
                train_root=str(base / "train_all"),
                test_root=str(base / "test"),
                output_dir=str(output_dir),
                val_ratio=None,
                overrides=[],
            )
            self.assertEqual(cmd_dataset_prepare(prepare_args), 0)
            audit_args = argparse.Namespace(
                config=str(config),
                index_dir=str(output_dir),
                strict=True,
                overrides=[],
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), self.assertRaises(RuntimeError):
                cmd_dataset_audit(audit_args)
            self.assertIn("content classification", buffer.getvalue())
            self.assertIn("shared content frames=0", buffer.getvalue())

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

    def test_prepare_if_missing_refuses_a_multi_process_launch(self) -> None:
        # cmd_train calls this before the distributed runtime exists, so all
        # eight ranks would otherwise scan and write the same index files.
        config = _prepare_if_missing_config(Path("nowhere"))
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "3"}, clear=False),
            self.assertRaises(SystemExit) as caught,
        ):
            _maybe_prepare_split(config)
        message = str(caught.exception)
        self.assertIn("WORLD_SIZE=8", message)
        self.assertIn("RANK=3", message)
        self.assertIn("cls-trainer dataset prepare", message)
        # The operator needs to know which artifacts were missing.
        self.assertIn("data.val_index=", message)

    def test_prepare_if_missing_is_a_noop_when_the_bundle_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = _prepare_if_missing_config(base)
            for path in config["data"].values():
                if isinstance(path, str) and path.startswith(str(base)):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text("stub", encoding="utf-8")
            self.assertEqual(_missing_split_artifacts(config["data"]), [])
            # Complete bundle: even eight ranks must pass straight through
            # without preparing anything.
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}, clear=False):
                _maybe_prepare_split(config)

    def test_prepare_if_missing_checks_every_artifact_not_just_val(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = _prepare_if_missing_config(base)
            data = config["data"]
            # A prepare that died after writing the frame parquets leaves the
            # video-entry indexes behind; checking val_index alone missed it.
            for key in ("train_index", "val_index", "test_index", "audit_path"):
                Path(data[key]).parent.mkdir(parents=True, exist_ok=True)
                Path(data[key]).write_text("stub", encoding="utf-8")
            missing = _missing_split_artifacts(data)
            self.assertEqual(
                sorted(item.split("=")[0] for item in missing),
                [
                    "data.test_video_index",
                    "data.train_video_index",
                    "data.val_video_index",
                ],
            )
            with (
                mock.patch.dict(
                    os.environ, {"WORLD_SIZE": "4", "RANK": "0"}, clear=False
                ),
                self.assertRaises(SystemExit),
            ):
                _maybe_prepare_split(config)

    def test_prepare_if_missing_needs_both_roots_single_process(self) -> None:
        config = _prepare_if_missing_config(Path("nowhere"))
        config["data"]["source_root"] = None
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}, clear=False),
            self.assertRaises(SystemExit) as caught,
        ):
            _maybe_prepare_split(config)
        self.assertIn("data.source_root", str(caught.exception))

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


class DatasetAnnotateCliTests(unittest.TestCase):
    """``dataset annotate`` CLI contract: input source, dedupe, merge.

    The index build is mocked out: these cover the annotate wiring (which
    input flags are legal, how mining top-K pairs collapse to sidecar rows and
    what a merge does to existing annotations), not index scanning.
    """

    def _config(self, base: Path) -> Path:
        # data.metadata_sidecar stays at its null default so the "no --out"
        # path is exercised for real.
        config = base / "annotate_config.yaml"
        config.write_text(
            """\
data:
  width: 448
  height: 208
  channels: 3
  frame_extensions: [".png"]
  split:
    mode: from_train
    val_ratio: 0.2
    seed: 20260728
pair:
  test_delta: 2
""",
            encoding="utf-8",
        )
        return config

    def _entries(self, uids: tuple[str, ...]) -> list:
        import numpy as np

        from game_cls.data.video_index import VideoEntry

        entries = []
        for uid in uids:
            game, video_id = uid.split("::")
            entries.append(
                VideoEntry(
                    game=game,
                    label=0,
                    video_id=video_id,
                    frame_ids=np.array([1, 2, 3]),
                    valid_start_positions={},
                    # The mock stands in for an indexed parquet, whose rows
                    # carry the persisted canonical uid (audit P0-5).
                    canonical_source_video_uid=uid,
                )
            )
        return entries

    def _annotate(
        self,
        config: Path,
        *,
        uids: tuple[str, ...] = ("game_a::01", "game_a::02", "game_a::03"),
        **overrides,
    ) -> tuple[int, str]:
        """Run cmd_dataset_annotate and capture its stdout."""
        import contextlib
        import io

        fields = {
            "config": str(config),
            "overrides": [],
            "metadata": None,
            "from_mining": None,
            "subtype": None,
            "out": None,
            "on_subtype_conflict": "refuse",
        }
        fields.update(overrides)
        args = argparse.Namespace(**fields)
        components = {
            "train_videos": self._entries(uids),
            "val_videos": [],
            "test_videos": None,
        }
        stream = io.StringIO()
        with (
            mock.patch(
                "game_cls.engine.training.loaders._build_real_data_components",
                return_value=components,
            ),
            contextlib.redirect_stdout(stream),
        ):
            code = cmd_dataset_annotate(args)
        return code, stream.getvalue()

    def _mining_manifest(self, path: Path) -> None:
        """Manifest where game_a::01 owns three top-K pairs, not one."""
        from game_cls.reports.benchmark import write_mining_manifest

        write_mining_manifest(
            [
                {
                    "source_video_uid": "game_a::01",
                    "p_positive": 0.71,
                    "rank_in_video": 1,
                    "frame0_id": 3,
                    "frame1_id": 5,
                },
                {
                    "source_video_uid": "game_a::01",
                    "p_positive": 0.94,
                    "rank_in_video": 0,
                    "frame0_id": 1,
                    "frame1_id": 3,
                },
                {
                    "source_video_uid": "game_a::01",
                    "p_positive": 0.55,
                    "rank_in_video": 2,
                    "frame0_id": 7,
                    "frame1_id": 9,
                },
                {
                    "source_video_uid": "game_a::02",
                    "p_positive": 0.63,
                    "rank_in_video": 0,
                    "frame0_id": 1,
                    "frame1_id": 3,
                },
            ],
            path,
        )

    def test_annotate_accepts_either_input_source_alone(self) -> None:
        # --metadata used to be required=True, so --from-mining could not be
        # used without also passing a meaningless --metadata.
        parser = build_parser()
        base = ["dataset", "annotate", "--config", "cfg.yaml"]
        mining = parser.parse_args(
            [*base, "--from-mining", "hard_negatives.parquet", "--subtype", "mined"]
        )
        self.assertIsNone(mining.metadata)
        self.assertEqual(mining.from_mining, "hard_negatives.parquet")
        metadata = parser.parse_args([*base, "--metadata", "meta.csv"])
        self.assertIsNone(metadata.from_mining)
        self.assertEqual(metadata.metadata, "meta.csv")

    def test_annotate_requires_exactly_one_input_source(self) -> None:
        parser = build_parser()
        base = ["dataset", "annotate", "--config", "cfg.yaml"]
        with (
            open(os.devnull, "w", encoding="utf-8") as devnull,
            mock.patch("sys.stderr", devnull),
        ):
            with self.assertRaises(SystemExit):
                parser.parse_args(base)  # neither
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [*base, "--metadata", "m.csv", "--from-mining", "m.parquet"]
                )

    def test_annotate_without_out_reports_the_missing_path(self) -> None:
        # data.metadata_sidecar is null, so the old Path(None) raised a
        # TypeError before this message could be printed.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mining = base / "hard_negatives.parquet"
            self._mining_manifest(mining)
            code, _ = self._annotate(
                self._config(base),
                from_mining=str(mining),
                subtype="mined_hard",
            )
            self.assertEqual(code, 2)

    def test_mining_rows_keep_one_highest_scoring_row_per_video(self) -> None:
        # _mining_rows owns the aggregation rule, so it is asserted directly:
        # end to end the primary key also survives because the merge is keyed
        # by uid, which would hide a regression here.
        rows = _mining_rows(
            [
                {"source_video_uid": "game_a::01", "p_positive": 0.71},
                {"source_video_uid": "game_a::01", "p_positive": 0.94},
                {"source_video_uid": "game_a::01", "p_positive": 0.55},
                {"source_video_uid": "game_a::02", "p_positive": 0.63},
            ],
            "mined_hard",
        )
        self.assertEqual(
            [row["source_video_uid"] for row in rows], ["game_a::01", "game_a::02"]
        )
        # The representative is the video's hardest pair, not the first seen.
        self.assertEqual(rows[0]["p_positive"], 0.94)
        self.assertEqual(rows[1]["p_positive"], 0.63)
        self.assertTrue(all(row["negative_subtype"] == "mined_hard" for row in rows))

    def test_annotate_from_mining_collapses_topk_pairs_per_video(self) -> None:
        from game_cls.data.sidecar import read_metadata_sidecar

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mining = base / "hard_negatives.parquet"
            self._mining_manifest(mining)
            sidecar = base / "sidecar.parquet"
            code, output = self._annotate(
                self._config(base),
                from_mining=str(mining),
                subtype="mined_hard",
                out=str(sidecar),
            )
            self.assertEqual(code, 0)
            payload = json.loads(output)
            # source_video_uid is the sidecar's primary key: four mined pairs
            # across two videos must land as exactly two rows, and the report
            # must show the aggregation rather than claim four videos.
            self.assertEqual(payload["mining_pairs"], 4)
            self.assertEqual(payload["videos"], 2)
            rows = read_metadata_sidecar(sidecar)
            self.assertEqual(sorted(rows), ["game_a::01", "game_a::02"])
            self.assertEqual(rows["game_a::01"]["negative_subtype"], "mined_hard")

    def test_annotate_merges_into_an_existing_sidecar(self) -> None:
        from game_cls.data.sidecar import read_metadata_sidecar, write_metadata_sidecar

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sidecar = base / "sidecar.parquet"
            # Hand-made annotations: one video the import never mentions, plus
            # a weight on a video the import does mention.
            write_metadata_sidecar(
                [
                    {
                        "source_video_uid": "game_a::03",
                        "negative_subtype": "flat_floor",
                        "scene_type": "indoor",
                        "sample_weight": 2.5,
                    },
                    {"source_video_uid": "game_a::01", "sample_weight": 3.5},
                ],
                sidecar,
            )
            mining = base / "hard_negatives.parquet"
            self._mining_manifest(mining)
            code, output = self._annotate(
                self._config(base),
                from_mining=str(mining),
                subtype="mined_hard",
                out=str(sidecar),
            )
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["videos"], 3)
            self.assertEqual(payload["new"], 1)  # game_a::02
            rows = read_metadata_sidecar(sidecar)
            # The untouched hand annotation survived the import.
            self.assertEqual(rows["game_a::03"]["negative_subtype"], "flat_floor")
            self.assertEqual(rows["game_a::03"]["scene_type"], "indoor")
            self.assertEqual(rows["game_a::03"]["sample_weight"], 2.5)
            # A mined video keeps its hand-set weight and gains the subtype.
            self.assertEqual(rows["game_a::01"]["negative_subtype"], "mined_hard")
            self.assertEqual(rows["game_a::01"]["sample_weight"], 3.5)

    def test_annotate_refuses_to_overwrite_a_different_subtype(self) -> None:
        from game_cls.data.sidecar import read_metadata_sidecar, write_metadata_sidecar

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sidecar = base / "sidecar.parquet"
            write_metadata_sidecar(
                [
                    {
                        "source_video_uid": "game_a::01",
                        "negative_subtype": "flat_floor",
                    }
                ],
                sidecar,
            )
            mining = base / "hard_negatives.parquet"
            self._mining_manifest(mining)
            config = self._config(base)
            code, output = self._annotate(
                config,
                from_mining=str(mining),
                subtype="mined_hard",
                out=str(sidecar),
            )
            self.assertEqual(code, 2)
            self.assertEqual(output, "")
            # Nothing was written: the sidecar still has only the hand row.
            rows = read_metadata_sidecar(sidecar)
            self.assertEqual(sorted(rows), ["game_a::01"])
            self.assertEqual(rows["game_a::01"]["negative_subtype"], "flat_floor")

    def test_annotate_subtype_conflict_keep_and_overwrite(self) -> None:
        from game_cls.data.sidecar import read_metadata_sidecar, write_metadata_sidecar

        for policy, expected in (("keep", "flat_floor"), ("overwrite", "mined_hard")):
            with (
                self.subTest(policy=policy),
                tempfile.TemporaryDirectory() as directory,
            ):
                base = Path(directory)
                sidecar = base / "sidecar.parquet"
                write_metadata_sidecar(
                    [
                        {
                            "source_video_uid": "game_a::01",
                            "negative_subtype": "flat_floor",
                        }
                    ],
                    sidecar,
                )
                mining = base / "hard_negatives.parquet"
                self._mining_manifest(mining)
                code, output = self._annotate(
                    self._config(base),
                    from_mining=str(mining),
                    subtype="mined_hard",
                    out=str(sidecar),
                    on_subtype_conflict=policy,
                )
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output)["subtype_conflicts"], 1)
                rows = read_metadata_sidecar(sidecar)
                self.assertEqual(rows["game_a::01"]["negative_subtype"], expected)

    def test_annotate_metadata_blank_cell_does_not_blank_the_sidecar(self) -> None:
        from game_cls.data.sidecar import read_metadata_sidecar, write_metadata_sidecar

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sidecar = base / "sidecar.parquet"
            write_metadata_sidecar(
                [
                    {
                        "source_video_uid": "game_a::01",
                        "scene_type": "indoor",
                        "sample_weight": 2.5,
                    }
                ],
                sidecar,
            )
            metadata = base / "meta.csv"
            metadata.write_text(
                "source_video_uid,negative_subtype,scene_type,sample_weight\n"
                "game_a::01,mined_hard,,\n",
                encoding="utf-8",
            )
            code, _ = self._annotate(
                self._config(base),
                metadata=str(metadata),
                out=str(sidecar),
            )
            self.assertEqual(code, 0)
            row = read_metadata_sidecar(sidecar)["game_a::01"]
            self.assertEqual(row["negative_subtype"], "mined_hard")
            # Empty CSV cells mean "not specified", not "erase this".
            self.assertEqual(row["scene_type"], "indoor")
            self.assertEqual(row["sample_weight"], 2.5)


if __name__ == "__main__":
    unittest.main()
