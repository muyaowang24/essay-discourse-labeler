from __future__ import annotations

import numpy as np

from essay_labeler.decoding import decode_word_level_predictions
from essay_labeler.postprocess import SpanThresholds, post_process_predictions


def predict_from_logits(
    dataframe: list[dict],
    logits: list[np.ndarray],
    word_id_batches: list[list[int]],
    labels: list[str],
    thresholds: SpanThresholds,
) -> list[dict]:
    all_predictions = []
    all_scores = []
    for row_logits, word_ids in zip(logits, word_id_batches):
        prediction_ids = row_logits.argmax(axis=-1).tolist()
        prediction_scores = row_logits.max(axis=-1).tolist()
        preds, scores = decode_word_level_predictions(
            prediction_ids=prediction_ids,
            prediction_scores=prediction_scores,
            word_ids=word_ids,
            labels=labels,
        )
        all_predictions.append(preds)
        all_scores.append(scores)
    return post_process_predictions(dataframe, all_predictions, all_scores, thresholds)
