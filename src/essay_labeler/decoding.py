from __future__ import annotations

from essay_labeler.labels import NON_LABEL, id_to_label


def decode_word_level_predictions(
    prediction_ids: list[int],
    prediction_scores: list[float],
    word_ids: list[int],
    labels: list[str],
) -> tuple[list[str], list[float]]:
    label_lookup = id_to_label(labels)
    word_predictions: list[str] = []
    word_scores: list[float] = []
    seen_word_ids: set[int] = set()
    for pred_id, score, word_id in zip(prediction_ids, prediction_scores, word_ids):
        if word_id == NON_LABEL or word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)
        word_predictions.append(label_lookup[pred_id])
        word_scores.append(float(score))
    return word_predictions, word_scores
