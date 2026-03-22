from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SpanThresholds:
    min_thresh: dict[str, int]
    prob_thresh: dict[str, float]


def post_process_predictions(
    dataframe: list[dict],
    predictions: list[list[str]],
    prediction_scores: list[list[float]],
    thresholds: SpanThresholds,
) -> list[dict]:
    final_predictions = []
    for row_idx, row in enumerate(dataframe):
        labels = predictions[row_idx]
        scores = prediction_scores[row_idx]
        cursor = 0
        while cursor < len(labels):
            current = labels[cursor]
            if current == "O":
                cursor += 1
                continue
            current = current.replace("B-", "I-")
            end = cursor + 1
            while end < len(labels) and labels[end].replace("B-", "I-") == current:
                end += 1
            avg_score = sum(scores[cursor:end]) / max(1, end - cursor)
            if end - cursor > thresholds.min_thresh.get(current, 0) and avg_score > thresholds.prob_thresh.get(
                current, 0.0
            ):
                final_predictions.append(
                    {
                        "id": row["id"],
                        "class": current.replace("I-", ""),
                        "new_predictionstring": " ".join(str(idx) for idx in range(cursor, end)),
                    }
                )
            cursor = end
    return final_predictions
