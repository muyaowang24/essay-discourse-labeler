from essay_labeler.postprocess import SpanThresholds, post_process_predictions


def test_postprocess_keeps_high_confidence_spans() -> None:
    frame = [{"id": "doc1"}]
    predictions = [["B-Lead", "I-Lead", "O"]]
    scores = [[0.9, 0.95, 0.1]]
    thresholds = SpanThresholds(min_thresh={"I-Lead": 1}, prob_thresh={"I-Lead": 0.5})
    result = post_process_predictions(frame, predictions, scores, thresholds)
    assert result == [
        {"id": "doc1", "class": "Lead", "new_predictionstring": "0 1"}
    ]
