import numpy as np

from essay_labeler.pipeline import predict_from_logits
from essay_labeler.postprocess import SpanThresholds


def test_predict_from_logits_formats_submission_rows() -> None:
    frame = [{"id": "doc1"}]
    logits = [
        np.array(
            [
                [0.1, 0.9, 0.0],
                [0.1, 0.0, 0.9],
                [0.9, 0.1, 0.0],
            ]
        )
    ]
    word_ids = [[0, 1, 2]]
    labels = ["O", "B-Lead", "I-Lead"]
    thresholds = SpanThresholds(min_thresh={"I-Lead": 1}, prob_thresh={"I-Lead": 0.5})
    result = predict_from_logits(frame, logits, word_ids, labels, thresholds)
    assert result == [
        {"id": "doc1", "class": "Lead", "new_predictionstring": "0 1"}
    ]
