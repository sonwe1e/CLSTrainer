from __future__ import annotations

from typing import Any

from ..contracts.evaluation import DecisionOutput, ErrorBatch, ErrorExtractor


def _threshold_band(probability: float) -> str | None:
    if 0.980 <= probability < 0.990:
        return "0.980-0.990"
    if 0.990 <= probability <= 0.995:
        return "0.990-0.995"
    return None


class BinaryErrorExtractor(ErrorExtractor):
    """Extracts FP / FN / near-threshold rows (USERPLAN §10.5).

    Produces the same row schema as the existing ``evaluate`` error path so the
    report files (``false_positive.parquet``, ``false_negative.parquet``,
    ``near_threshold.parquet``, ``errors.html``) stay compatible.
    """

    def __init__(self, quick_error_limit: int = 200) -> None:
        self._quick_limit = quick_error_limit

    def extract(
        self, prediction_batch: Any, decision: DecisionOutput, checkpoint_step: int
    ) -> ErrorBatch:
        targets = prediction_batch.targets.detach().cpu().tolist()
        margins = decision.auxiliary["margins"].float().detach().cpu().tolist()
        probabilities = decision.scores.float().detach().cpu().tolist()
        predictions = decision.predictions.detach().cpu().tolist()
        logits = prediction_batch.extras["logits"].float().detach().cpu()
        metadata = prediction_batch.metadata

        false_positives: list[dict] = []
        false_negatives: list[dict] = []
        near_threshold: list[dict] = []

        for index, meta in enumerate(metadata):
            label = int(meta.get("label", targets[index]))
            prediction = int(predictions[index])
            logit = logits[index]
            row = {
                **meta,
                "label": label,
                "logit0": float(logit[0]),
                "logit1": float(logit[1]),
                "margin": margins[index],
                "probability_class1": probabilities[index],
                "prediction": prediction,
                "checkpoint_step": checkpoint_step,
            }
            if prediction != label:
                error_type = "FP" if prediction else "FN"
                entry = {**row, "error_type": error_type}
                if error_type == "FP":
                    false_positives.append(entry)
                else:
                    false_negatives.append(entry)
            band = _threshold_band(probabilities[index])
            if band is not None:
                near_threshold.append({**row, "threshold_band": band})

        if self._quick_limit > 0:
            false_positives = false_positives[: self._quick_limit]
            false_negatives = false_negatives[: self._quick_limit]
            near_threshold = near_threshold[: self._quick_limit]

        return ErrorBatch(
            false_positives=false_positives,
            false_negatives=false_negatives,
            near_threshold=near_threshold,
        )
