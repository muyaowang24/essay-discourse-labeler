from essay_labeler.metrics import calc_overlap, macro_f1_score


def test_calc_overlap_returns_expected_ratios() -> None:
    overlap1, overlap2 = calc_overlap("1 2 3", "2 3 4")
    assert round(overlap1, 2) == 0.67
    assert round(overlap2, 2) == 0.67


def test_macro_f1_score_is_one_for_perfect_predictions() -> None:
    pred = [{"id": "a", "class": "Lead", "new_predictionstring": "0 1"}]
    gt = [{"id": "a", "discourse_type": "Lead", "new_predictionstring": "0 1"}]
    assert macro_f1_score(pred, gt) == 1.0
