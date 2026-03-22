from __future__ import annotations

def calc_overlap(prediction: str, ground_truth: str) -> tuple[float, float]:
    pred_tokens = set(str(prediction).split())
    gt_tokens = set(str(ground_truth).split())
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0
    intersection = len(pred_tokens & gt_tokens)
    return intersection / len(gt_tokens), intersection / len(pred_tokens)


def score_feedback_comp(pred_df: list[dict], gt_df: list[dict]) -> float:
    pred_rows = [
        {"pred_id": idx, "id": row["id"], "class": row["class"], "new_predictionstring": row["new_predictionstring"]}
        for idx, row in enumerate(pred_df)
    ]
    gt_rows = [
        {
            "gt_id": idx,
            "id": row["id"],
            "discourse_type": row["discourse_type"],
            "new_predictionstring": row["new_predictionstring"],
        }
        for idx, row in enumerate(gt_df)
    ]

    candidates = []
    for pred in pred_rows:
        for gt in gt_rows:
            if pred["id"] != gt["id"] or pred["class"] != gt["discourse_type"]:
                continue
            overlap1, overlap2 = calc_overlap(pred["new_predictionstring"], gt["new_predictionstring"])
            if overlap1 >= 0.5 and overlap2 >= 0.5:
                candidates.append(
                    {
                        "pred_id": pred["pred_id"],
                        "gt_id": gt["gt_id"],
                        "max_overlap": max(overlap1, overlap2),
                    }
                )
    candidates.sort(key=lambda row: row["max_overlap"], reverse=True)
    used_pred = set()
    used_gt = set()
    tp = 0
    for candidate in candidates:
        if candidate["pred_id"] in used_pred or candidate["gt_id"] in used_gt:
            continue
        used_pred.add(candidate["pred_id"])
        used_gt.add(candidate["gt_id"])
        tp += 1
    fp = len(pred_rows) - tp
    fn = len(gt_rows) - tp
    denominator = tp + 0.5 * (fp + fn)
    return 0.0 if denominator == 0 else tp / denominator


def macro_f1_score(pred_df: list[dict], gt_df: list[dict]) -> float:
    classes = sorted({row["discourse_type"] for row in gt_df if row.get("discourse_type")})
    scores = []
    for discourse_type in classes:
        pred_slice = [row for row in pred_df if row["class"] == discourse_type]
        gt_slice = [row for row in gt_df if row["discourse_type"] == discourse_type]
        scores.append(score_feedback_comp(pred_slice, gt_slice))
    return float(sum(scores) / len(scores)) if scores else 0.0
