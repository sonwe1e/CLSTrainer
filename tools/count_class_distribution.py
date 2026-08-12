#!/usr/bin/env python3
"""Count class/game balance for either the image or video backend.

For online train delta, ``samples`` means indexed start positions.  Each start
produces exactly one pair per epoch and delta is sampled online, so this is the
correct frequency basis for class weighting/Focal alpha.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from clstrainer_lite.config import load_config
from clstrainer_lite.data import build_datasets


@dataclass
class Row:
    split: str
    game: str
    label: int
    videos: int = 0
    frames: int = 0
    samples: int = 0


def rows_for_dataset(dataset, split_name: str | None = None) -> list[Row]:
    grouped: dict[tuple[str, int], Row] = {}
    for index, video in enumerate(dataset.videos):
        key = (video.game, video.label)
        row = grouped.setdefault(
            key, Row(split_name or dataset.split_name, video.game, video.label)
        )
        row.videos += 1
        row.frames += int(video.frame_count)
        row.samples += int(dataset.position_count_for_video(index))
    return [grouped[key] for key in sorted(grouped)]


def label_totals(rows: list[Row]) -> dict[int, int]:
    result = {0: 0, 1: 0}
    for row in rows:
        result[row.label] += row.samples
    return result


def recommended_alpha(counts: dict[int, int]) -> list[float]:
    if counts[0] <= 0 or counts[1] <= 0:
        raise ValueError("actual_train must contain both class 0 and class 1 samples")
    total = counts[0] + counts[1]
    raw = {label: math.sqrt(total / counts[label]) for label in (0, 1)}
    weighted_mean = sum(counts[label] * raw[label] for label in (0, 1)) / total
    return [raw[0] / weighted_mean, raw[1] / weighted_mean]


def print_table(rows: list[Row]) -> None:
    headers = ("split", "game", "label", "videos", "frames", "samples")
    values = [
        (r.split, r.game, str(r.label), str(r.videos), str(r.frames), str(r.samples))
        for r in rows
    ]
    widths = [len(x) for x in headers]
    for row in values:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    fmt = lambda row: "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
    print(fmt(headers))
    print("  ".join("-" * width for width in widths))
    for row in values:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="dataset_stats")
    args = parser.parse_args()

    config = load_config(args.config)
    train, val, test = build_datasets(config)
    split_rows = {
        "actual_train": rows_for_dataset(train, "actual_train"),
        "val": rows_for_dataset(val),
        "test": rows_for_dataset(test),
    }
    rows = [row for group in split_rows.values() for row in group]
    print_table(rows)

    counts = label_totals(split_rows["actual_train"])
    alpha = recommended_alpha(counts)
    ratio = max(counts.values()) / min(counts.values())
    print("\nactual_train sampling-position counts:", counts)
    print(f"imbalance max/min: {ratio:.3f}x")
    print(f"recommended focal: gamma=1.5 alpha=[{alpha[0]:.6f}, {alpha[1]:.6f}]")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "class_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)

    per_game: dict[tuple[str, str], dict[int, Row]] = defaultdict(dict)
    for row in rows:
        per_game[(row.split, row.game)][row.label] = row
    game_rows = []
    for (split, game), classes in sorted(per_game.items()):
        zero = classes.get(0, Row(split, game, 0))
        one = classes.get(1, Row(split, game, 1))
        game_rows.append(
            {
                "split": split,
                "game": game,
                "class0_samples": zero.samples,
                "class1_samples": one.samples,
                "sample_imbalance_max_min": (
                    max(zero.samples, one.samples) / min(zero.samples, one.samples)
                    if min(zero.samples, one.samples) > 0
                    else None
                ),
            }
        )
    with (output / "game_balance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "game",
                "class0_samples",
                "class1_samples",
                "sample_imbalance_max_min",
            ],
        )
        writer.writeheader()
        writer.writerows(game_rows)

    recommendation = {
        "basis": "actual_train online sampling positions",
        "backend": config["data"]["backend"],
        "train_delta_range": config["data"]["train_delta_range"],
        "train_delta_probabilities": config["data"].get("train_delta_probabilities"),
        "eval_delta": config["data"]["eval_delta"],
        "sample_counts": {str(k): v for k, v in counts.items()},
        "recommended": {"type": "focal", "gamma": 1.5, "alpha": alpha},
    }
    (output / "focal_loss_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("written:", output.resolve())


if __name__ == "__main__":
    main()
