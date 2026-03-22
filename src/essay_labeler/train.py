from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from essay_labeler.config import ExperimentConfig
from essay_labeler.data import build_training_frame, load_essays
from essay_labeler.dataset import EssayDataset
from essay_labeler.metrics import macro_f1_score
from essay_labeler.modeling import build_model, build_tokenizer
from essay_labeler.pipeline import predict_from_logits
from essay_labeler.postprocess import SpanThresholds
from essay_labeler.utils import ensure_dir, get_device, seed_everything


def _active_logits(raw_logits: torch.Tensor, word_ids: torch.Tensor, num_labels: int) -> torch.Tensor:
    flat_word_ids = word_ids.view(-1)
    active_mask = flat_word_ids.unsqueeze(1).expand(flat_word_ids.shape[0], num_labels) != -1
    logits = raw_logits.view(-1, num_labels)
    return torch.masked_select(logits, active_mask).view(-1, num_labels)


def _active_labels(labels: torch.Tensor) -> torch.Tensor:
    active_mask = labels.view(-1) != -100
    return torch.masked_select(labels.view(-1), active_mask)


def _prediction_batches(model, dataloader, device) -> tuple[list, list]:
    logits_batches = []
    word_id_batches = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            raw_logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            for row_logits, row_word_ids in zip(raw_logits, batch["word_ids"]):
                logits_batches.append(torch.softmax(row_logits, dim=-1).cpu().numpy())
                word_id_batches.append(row_word_ids.tolist())
    return logits_batches, word_id_batches


def train(config: ExperimentConfig) -> list[dict]:
    seed_everything(config.seed)
    device = get_device()
    train_frame, annotations = build_training_frame(config)
    tokenizer = build_tokenizer(config)
    thresholds = SpanThresholds(
        min_thresh=config.postprocess.min_thresh,
        prob_thresh=config.postprocess.prob_thresh,
    )
    checkpoints_dir = ensure_dir(config.data.artifacts_dir / "checkpoints")
    records: list[dict] = []

    for fold in range(config.training.folds):
        model = build_model(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        train_split = [row for row in train_frame if row["fold"] != fold]
        valid_split = [row for row in train_frame if row["fold"] == fold]
        valid_ids = {row["id"] for row in valid_split}
        valid_annotations = [row for row in annotations if row["id"] in valid_ids]

        train_loader = DataLoader(
            EssayDataset(
                train_split,
                tokenizer=tokenizer,
                max_length=config.training.max_length,
                labels=config.task.labels,
                has_labels=True,
            ),
            batch_size=config.training.train_batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
        )
        valid_loader = DataLoader(
            EssayDataset(
                valid_split,
                tokenizer=tokenizer,
                max_length=config.training.max_length,
                labels=config.task.labels,
                has_labels=True,
            ),
            batch_size=config.training.valid_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )

        best_score = -1.0
        best_path = checkpoints_dir / f"fold_{fold}" / "model.pt"
        ensure_dir(best_path.parent)

        for _epoch in range(config.training.epochs):
            model.train()
            progress = tqdm(train_loader, desc=f"fold {fold} train", leave=False)
            for batch in progress:
                optimizer.zero_grad()
                raw_logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                logits = _active_logits(raw_logits, batch["word_ids"].to(device), config.num_labels)
                labels = _active_labels(batch["labels"].to(device))
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            valid_predictions = evaluate_model(
                model=model,
                dataframe=valid_split,
                dataloader=valid_loader,
                labels=config.task.labels,
                thresholds=thresholds,
                device=device,
            )
            score = macro_f1_score(valid_predictions, valid_annotations)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)

        records.append({"fold": fold, "macro_f1": best_score, "checkpoint": str(best_path)})
    _write_csv(config.data.artifacts_dir / "training_metrics.csv", records, ["fold", "macro_f1", "checkpoint"])
    return records


def evaluate_model(model, dataframe, dataloader, labels, thresholds, device) -> list[dict]:
    logits_batches, word_id_batches = _prediction_batches(model, dataloader, device)
    return predict_from_logits(dataframe, logits_batches, word_id_batches, labels, thresholds)


def evaluate(config: ExperimentConfig, checkpoint: str | Path) -> dict[str, float]:
    seed_everything(config.seed)
    device = get_device()
    train_frame, annotations = build_training_frame(config)
    tokenizer = build_tokenizer(config)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    thresholds = SpanThresholds(
        min_thresh=config.postprocess.min_thresh,
        prob_thresh=config.postprocess.prob_thresh,
    )
    fold = _checkpoint_fold(checkpoint)
    valid_split = [row for row in train_frame if row["fold"] == fold]
    valid_ids = {row["id"] for row in valid_split}
    valid_annotations = [row for row in annotations if row["id"] in valid_ids]
    valid_loader = DataLoader(
        EssayDataset(
            valid_split,
            tokenizer=tokenizer,
            max_length=config.training.inference_max_length,
            labels=config.task.labels,
            has_labels=True,
        ),
        batch_size=config.training.valid_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    predictions = evaluate_model(
        model=model,
        dataframe=valid_split,
        dataloader=valid_loader,
        labels=config.task.labels,
        thresholds=thresholds,
        device=device,
    )
    score = macro_f1_score(predictions, valid_annotations)
    output_path = config.data.artifacts_dir / "evaluation_predictions.csv"
    _write_csv(output_path, predictions, ["id", "class", "new_predictionstring"])
    return {"macro_f1": score, "predictions_path": str(output_path)}


def predict(config: ExperimentConfig, checkpoints: list[str], input_dir: str | Path, output: str | Path) -> Path:
    seed_everything(config.seed)
    device = get_device()
    essays = load_essays(input_dir)
    tokenizer = build_tokenizer(config)
    dataset = FeedbackPrizeDataset(
        essays,
        tokenizer=tokenizer,
        max_length=config.training.inference_max_length,
        labels=config.task.labels,
        has_labels=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.valid_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    thresholds = SpanThresholds(
        min_thresh=config.postprocess.min_thresh,
        prob_thresh=config.postprocess.prob_thresh,
    )

    ensemble_logits = None
    word_id_batches = None
    for checkpoint in checkpoints:
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        logits_batches, current_word_ids = _prediction_batches(model, dataloader, device)
        stacked = [batch.copy() for batch in logits_batches]
        if ensemble_logits is None:
            ensemble_logits = stacked
            word_id_batches = current_word_ids
        else:
            ensemble_logits = [left + right for left, right in zip(ensemble_logits, stacked)]
    averaged_logits = [batch / len(checkpoints) for batch in ensemble_logits or []]
    predictions = predict_from_logits(
        dataframe=essays,
        logits=averaged_logits,
        word_id_batches=word_id_batches or [],
        labels=config.task.labels,
        thresholds=thresholds,
    )
    final_output = Path(output)
    ensure_dir(final_output.parent)
    _write_csv(
        final_output,
        [
            {"id": row["id"], "class": row["class"], "predictionstring": row["new_predictionstring"]}
            for row in predictions
        ],
        ["id", "class", "predictionstring"],
    )
    return final_output


def _checkpoint_fold(checkpoint: str | Path) -> int:
    checkpoint_path = Path(checkpoint)
    for part in checkpoint_path.parts:
        if part.startswith("fold_"):
            return int(part.split("_", maxsplit=1)[1])
    raise ValueError(f"Unable to determine fold from checkpoint path: {checkpoint}")


def _write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
