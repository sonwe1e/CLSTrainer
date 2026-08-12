from __future__ import annotations

import csv
import json
from pathlib import Path


HISTORY_FIELDS = [
    "step",
    "epoch",
    "learning_rate",
    "train_loss",
    "train_f1",
    "train_samples",
    "val_loss",
    "val_f1",
    "val_samples",
    "val_class0_f1",
    "val_class1_f1",
    "val_class0_recall",
    "val_class1_recall",
    "test_loss",
    "test_f1",
    "test_samples",
    "test_class0_f1",
    "test_class1_f1",
    "test_class0_recall",
    "test_class1_recall",
]


def write_history(history: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def write_metric_details(evaluations: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_detail.json").write_text(
        json.dumps(evaluations, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    points = [(int(row["step"]), row.get(key)) for row in history]
    points = [(step, float(value)) for step, value in points if value is not None]
    return [step for step, _ in points], [value for _, value in points]


def plot_history(history: list[dict], output_dir: str | Path) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    plt.figure(figsize=(7.6, 4.6))
    for split in ("train", "val", "test"):
        steps, values = _series(history, f"{split}_loss")
        if steps:
            plt.plot(steps, values, marker="o", markersize=3, label=split)
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.title("Train / validation / test loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7.6, 4.6))
    for split in ("train", "val", "test"):
        steps, values = _series(history, f"{split}_f1")
        if steps:
            plt.plot(steps, values, marker="o", markersize=3, label=split)
    plt.xlabel("Optimizer step")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.0)
    plt.title("Train / validation / test F1")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "f1_curve.png", dpi=150)
    plt.close()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for split in ("val", "test"):
        for class_id in (0, 1):
            steps, values = _series(history, f"{split}_class{class_id}_f1")
            if steps:
                axes[0].plot(steps, values, marker="o", markersize=3, label=f"{split} class {class_id}")
            steps, values = _series(history, f"{split}_class{class_id}_recall")
            if steps:
                axes[1].plot(steps, values, marker="o", markersize=3, label=f"{split} class {class_id}")
    for ax, name in zip(axes, ("Per-class F1", "Per-class recall"), strict=True):
        ax.set_xlabel("Optimizer step")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "class_metrics_curve.png", dpi=150)
    plt.close(figure)


def plot_diagnostics(report: dict, path: str | Path, *, title: str) -> None:
    """Render one compact diagnosis page from an aggregated evaluation report."""
    per_game = report.get("per_game") or {}
    if not per_game:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    games = list(per_game)
    x = np.arange(len(games), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    for ax, metric, name in (
        (axes[0, 0], "f1", "F1 by class"),
        (axes[0, 1], "recall", "Recall by class"),
        (axes[1, 0], "precision", "Precision by class"),
    ):
        class0 = [float(per_game[game]["per_class"]["0"][metric]) for game in games]
        class1 = [float(per_game[game]["per_class"]["1"][metric]) for game in games]
        ax.bar(x - width / 2, class0, width, label="class 0")
        ax.bar(x + width / 2, class1, width, label="class 1")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(name)
        ax.set_xticks(x, games, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()

    fp = [int(per_game[game]["fp"]) for game in games]
    fn = [int(per_game[game]["fn"]) for game in games]
    axes[0, 2].bar(x - width / 2, fp, width, label="FP: 0→1")
    axes[0, 2].bar(x + width / 2, fn, width, label="FN: 1→0")
    axes[0, 2].set_title("Misclassification counts")
    axes[0, 2].set_xticks(x, games, rotation=25, ha="right")
    axes[0, 2].grid(axis="y", alpha=0.2)
    axes[0, 2].legend()

    histogram = report.get("score_histogram") or {}
    bins = int(histogram.get("bins", 20))
    centers = (np.arange(bins) + 0.5) / bins
    class0_hist = np.asarray(histogram.get("class0", [0] * bins), dtype=float)
    class1_hist = np.asarray(histogram.get("class1", [0] * bins), dtype=float)
    if class0_hist.sum() > 0:
        class0_hist /= class0_hist.sum()
    if class1_hist.sum() > 0:
        class1_hist /= class1_hist.sum()
    axes[1, 1].plot(centers, class0_hist, marker="o", markersize=3, label="true class 0")
    axes[1, 1].plot(centers, class1_hist, marker="o", markersize=3, label="true class 1")
    threshold = float(report.get("threshold", 0.5) or 0.5)
    axes[1, 1].axvline(threshold, linestyle="--", linewidth=1, label=f"threshold={threshold:.3f}")
    axes[1, 1].set_xlim(0.0, 1.0)
    axes[1, 1].set_title("Predicted P(class=1)")
    axes[1, 1].set_xlabel("P(class=1)")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend()

    confidence = report.get("confidence") or {}
    names = ["tp", "tn", "fp", "fn"]
    means = [
        0.0 if confidence.get(name, {}).get("mean") is None else float(confidence[name]["mean"])
        for name in names
    ]
    axes[1, 2].bar(names, means)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Prediction confidence by outcome")
    axes[1, 2].set_ylabel("mean max-class probability")
    axes[1, 2].grid(axis="y", alpha=0.2)
    for index, name in enumerate(names):
        count = int(confidence.get(name, {}).get("count", 0))
        axes[1, 2].text(index, means[index] + 0.02, f"n={count}", ha="center", fontsize=8)

    overall = report.get("per_class", {})
    subtitle = (
        f"overall F1={float(report.get('f1', 0.0)):.4f}  "
        f"class0 F1={float(overall.get('0', {}).get('f1', 0.0)):.4f}  "
        f"class1 F1={float(overall.get('1', {}).get('f1', 0.0)):.4f}"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(Path(path), dpi=150)
    plt.close(figure)
