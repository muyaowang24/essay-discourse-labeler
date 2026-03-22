from __future__ import annotations

import csv
from pathlib import Path

from sklearn.model_selection import KFold

from essay_labeler.config import ExperimentConfig


def load_essays(directory: str | Path) -> list[dict]:
    essay_dir = Path(directory)
    rows = []
    for path in sorted(essay_dir.glob("*.txt")):
        text = path.read_text()
        rows.append({"id": path.stem, "text": text, "text_split": text.split()})
    return rows


def load_annotations(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    for row in rows:
        if not row.get("new_predictionstring") and row.get("predictionstring"):
            row["new_predictionstring"] = row["predictionstring"]
    return rows


def build_entities_frame(essays: list[dict], annotations: list[dict]) -> list[dict]:
    grouped_annotations: dict[str, list[dict]] = {}
    for annotation in annotations:
        grouped_annotations.setdefault(annotation["id"], []).append(annotation)
    rows = []
    for row in essays:
        token_count = len(row["text_split"])
        entities = ["O"] * token_count
        matches = grouped_annotations.get(row["id"])
        if matches is not None:
            for ann in matches:
                token_ids = [int(x) for x in str(ann["new_predictionstring"]).split() if x]
                if not token_ids:
                    continue
                first = token_ids[0]
                if first < token_count:
                    entities[first] = f"B-{ann['discourse_type']}"
                for token_id in token_ids[1:]:
                    if token_id < token_count:
                        entities[token_id] = f"I-{ann['discourse_type']}"
        rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "text_split": row["text_split"],
                "entities": entities,
            }
        )
    return rows


def assign_folds(frame: list[dict], folds: int, seed: int) -> list[dict]:
    result = [dict(row) for row in frame]
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(range(len(result)))):
        for idx in valid_idx:
            result[idx]["fold"] = fold
    return result


def build_training_frame(config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    annotation_path = config.data.corrected_csv or config.data.train_csv
    annotations = load_annotations(annotation_path)
    essays = load_essays(config.data.train_dir)
    training_frame = build_entities_frame(essays, annotations)
    training_frame = assign_folds(training_frame, config.training.folds, config.seed)
    return training_frame, annotations
