from __future__ import annotations

from typing import Any

from ..contracts.evaluation import DecisionOutput, GroupAggregator
from ..metrics.binary_metrics import metrics_from_counts


def _counter(margin: float, target: int, cutoff: float) -> list[int]:
    prediction = margin > cutoff
    return [
        int(prediction and target == 1),
        int(prediction and target == 0),
        int(not prediction and target == 1),
        int(not prediction and target == 0),
    ]


class BinaryGroupAggregator(GroupAggregator):
    """Aggregates metrics by game, game+label, and video (sequence).

    Reads group keys from the prediction batch metadata's ``groups`` mapping,
    preserving the existing ``metrics_by_game`` / ``metrics_by_game_label`` /
    ``metrics_by_video`` outputs (USERPLAN §10.4).
    """

    def __init__(self) -> None:
        self._game: dict = {}
        self._game_label: dict = {}
        self._video: dict = {}
        self._video_confidence: dict = {}

    def update(self, prediction_batch: Any, decision: DecisionOutput) -> None:
        import torch

        targets = prediction_batch.targets.cpu().tolist()
        margins = decision.auxiliary["margins"].float().cpu().tolist()
        probabilities = decision.scores.float().cpu().tolist()
        cutoff = decision.auxiliary["cutoff"]
        metadata = prediction_batch.metadata
        for index, meta in enumerate(metadata):
            game = str(meta.get("game", "unknown"))
            label = int(meta.get("label", targets[index]))
            video_id = str(meta.get("video_id", meta.get("sequence_id", "unknown")))
            counts = _counter(margins[index], targets[index], cutoff)
            self._game.setdefault(game, [0, 0, 0, 0])
            for i, value in enumerate(counts):
                self._game[game][i] += value
            self._game_label.setdefault((game, label), [0, 0, 0, 0])
            for i, value in enumerate(counts):
                self._game_label[(game, label)][i] += value
            self._video.setdefault((game, label, video_id), [0, 0, 0, 0])
            for i, value in enumerate(counts):
                self._video[(game, label, video_id)][i] += value
            conf = self._video_confidence.setdefault(
                (game, label, video_id), {"count": 0, "sum": 0.0, "min": 1.0, "max": 0.0}
            )
            conf["count"] += 1
            conf["sum"] += probabilities[index]
            conf["min"] = min(conf["min"], probabilities[index])
            conf["max"] = max(conf["max"], probabilities[index])

    def distributed_reduce(self, runtime: Any) -> None:
        # Group aggregation merges gathered dicts on rank 0 in compute().
        del runtime

    def compute(self) -> dict[str, Any]:
        by_game = []
        for game, counts in sorted(self._game.items()):
            metrics = metrics_from_counts(*counts).to_dict()
            by_game.append({"game": game, **metrics})
        by_game_label = []
        for (game, label), counts in sorted(self._game_label.items()):
            metrics = metrics_from_counts(*counts).to_dict()
            by_game_label.append({"game": game, "label": label, **metrics})
        by_video = []
        for (game, label, video_id), counts in sorted(self._video.items()):
            metrics = metrics_from_counts(*counts).to_dict()
            conf = self._video_confidence[(game, label, video_id)]
            by_video.append(
                {
                    "game": game,
                    "label": label,
                    "video_id": video_id,
                    **metrics,
                    "mean_probability_class1": conf["sum"] / conf["count"] if conf["count"] else 0.0,
                    "min_probability_class1": conf["min"],
                    "max_probability_class1": conf["max"],
                }
            )
        return {"by_game": by_game, "by_game_label": by_game_label, "by_video": by_video}
