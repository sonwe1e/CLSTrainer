from __future__ import annotations

import math
from collections import Counter

import torch

from clstrainer_lite.data import ImagePairDataset, ImageVideo
from clstrainer_lite.metrics import DiagnosticAccumulator
from clstrainer_lite.sampler import BalancedVideoSampler
from clstrainer_lite.video_data import VideoPairDataset, VideoRecord


def _image_dataset() -> ImagePairDataset:
    videos = []
    for game, frames in (("game_a", 25), ("game_b", 9)):
        for label in (0, 1):
            ids = tuple(range(frames))
            videos.append(
                ImageVideo(
                    game=game,
                    label=label,
                    video_id=f"{game}_{label}",
                    frame_ids=ids,
                    paths=tuple(f"unused_{game}_{label}_{i}.png" for i in ids),
                )
            )
    return ImagePairDataset(
        videos,
        source_root=".",
        split_name="train",
        delta=[1, 2],
        delta_probabilities=[0.5, 0.5],
        online_delta=True,
        image_size=(8, 8),
    )


def _video_dataset() -> VideoPairDataset:
    videos = []
    for game, frames in (("game_a", 25), ("game_b", 9)):
        for label in (0, 1):
            videos.append(
                VideoRecord(
                    game=game,
                    label=label,
                    video_id=f"{game}_{label}",
                    path=f"unused_{game}_{label}.mp4",
                    frame_count=frames,
                    fps_num=30,
                    fps_den=1,
                    width=8,
                    height=8,
                )
            )
    return VideoPairDataset(
        videos,
        source_root=".",
        split_name="train",
        delta=[1, 2],
        delta_probabilities=[0.5, 0.5],
        online_delta=True,
        image_size=(8, 8),
        chunk_frames=4,
    )


def _assert_balanced(dataset) -> None:
    sampler = BalancedVideoSampler(
        dataset,
        class_probability=[0.5, 0.5],
        game_balance_alpha=1.0,
        samples_per_epoch=12_000,
        seed=123,
    )
    counts = Counter()
    for index in sampler:
        _, video, _ = dataset._locate(index)
        counts[(video.game, video.label)] += 1
    total = sum(counts.values())
    for game in ("game_a", "game_b"):
        game_fraction = sum(counts[(game, label)] for label in (0, 1)) / total
        assert 0.47 <= game_fraction <= 0.53
    class1_fraction = sum(counts[(game, 1)] for game in ("game_a", "game_b")) / total
    assert 0.47 <= class1_fraction <= 0.53


def test_balanced_sampler_is_backend_agnostic():
    _assert_balanced(_image_dataset())
    _assert_balanced(_video_dataset())


def test_balanced_sampler_ddp_has_equal_rank_lengths():
    dataset = _image_dataset()
    samplers = [
        BalancedVideoSampler(dataset, samples_per_epoch=101, rank=rank, world_size=8, seed=5)
        for rank in range(8)
    ]
    assert {len(sampler) for sampler in samplers} == {math.ceil(101 / 8)}


def test_diagnostic_accumulator_reports_games_classes_and_confidence():
    accumulator = DiagnosticAccumulator(torch.device("cpu"), ["game_a", "game_b"], bins=20)
    probabilities = torch.tensor([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4])
    logits = torch.stack((torch.log1p(-probabilities), torch.log(probabilities)), dim=1)
    targets = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    games = ["game_a"] * 4 + ["game_b"] * 4
    accumulator.update(torch.tensor(0.5), logits, targets, games, 0.5)
    report = accumulator.compute()

    assert report["samples"] == 8
    assert (report["tp"], report["fp"], report["fn"], report["tn"]) == (3, 1, 1, 3)
    assert abs(report["per_class"]["1"]["f1"] - 0.75) < 1e-8
    assert abs(report["per_class"]["0"]["f1"] - 0.75) < 1e-8
    assert report["per_game"]["game_a"]["fp"] == 1
    assert report["per_game"]["game_a"]["fn"] == 1
    assert report["per_game"]["game_b"]["fp"] == 0
    assert report["per_game"]["game_b"]["fn"] == 0
    assert abs(report["confidence"]["fp"]["mean"] - 0.8) < 1e-6
    assert abs(report["confidence"]["fn"]["mean"] - 0.8) < 1e-6
    assert sum(report["score_histogram"]["class0"]) == 4
    assert sum(report["score_histogram"]["class1"]) == 4
