from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from essay_labeler.labels import DEFAULT_LABELS


@dataclass(slots=True)
class TaskConfig:
    labels: list[str]


@dataclass(slots=True)
class DataConfig:
    train_dir: Path
    test_dir: Path
    train_csv: Path
    corrected_csv: Path | None
    artifacts_dir: Path


@dataclass(slots=True)
class ModelConfig:
    backbone_name: str
    tokenizer_name: str | None
    dropout_prob: float
    multi_sample_dropout: int


@dataclass(slots=True)
class TrainingConfig:
    folds: int
    epochs: int
    train_batch_size: int
    valid_batch_size: int
    learning_rate: float
    weight_decay: float
    max_length: int
    inference_max_length: int
    num_workers: int
    grad_accumulation_steps: int
    save_best_only: bool


@dataclass(slots=True)
class PostProcessConfig:
    min_thresh: dict[str, int]
    prob_thresh: dict[str, float]


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    task: TaskConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    postprocess: PostProcessConfig

    @property
    def num_labels(self) -> int:
        return len(self.task.labels)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            pass
    return raw


def apply_overrides(config_dict: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config_dict
    merged = dict(config_dict)
    for override in overrides:
        key, value = override.split("=", maxsplit=1)
        target = merged
        path = key.split(".")
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = _coerce_value(value)
    return merged


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    merged = apply_overrides(raw, overrides)
    labels = merged.get("task", {}).get("labels") or DEFAULT_LABELS

    return ExperimentConfig(
        seed=int(merged["seed"]),
        task=TaskConfig(labels=list(labels)),
        data=DataConfig(
            train_dir=Path(merged["data"]["train_dir"]),
            test_dir=Path(merged["data"]["test_dir"]),
            train_csv=Path(merged["data"]["train_csv"]),
            corrected_csv=Path(merged["data"]["corrected_csv"])
            if merged["data"].get("corrected_csv")
            else None,
            artifacts_dir=Path(merged["data"]["artifacts_dir"]),
        ),
        model=ModelConfig(
            backbone_name=merged["model"]["backbone_name"],
            tokenizer_name=merged["model"].get("tokenizer_name"),
            dropout_prob=float(merged["model"]["dropout_prob"]),
            multi_sample_dropout=int(merged["model"]["multi_sample_dropout"]),
        ),
        training=TrainingConfig(
            folds=int(merged["training"]["folds"]),
            epochs=int(merged["training"]["epochs"]),
            train_batch_size=int(merged["training"]["train_batch_size"]),
            valid_batch_size=int(merged["training"]["valid_batch_size"]),
            learning_rate=float(merged["training"]["learning_rate"]),
            weight_decay=float(merged["training"]["weight_decay"]),
            max_length=int(merged["training"]["max_length"]),
            inference_max_length=int(merged["training"]["inference_max_length"]),
            num_workers=int(merged["training"]["num_workers"]),
            grad_accumulation_steps=int(merged["training"]["grad_accumulation_steps"]),
            save_best_only=bool(merged["training"]["save_best_only"]),
        ),
        postprocess=PostProcessConfig(
            min_thresh={k: int(v) for k, v in merged["postprocess"]["min_thresh"].items()},
            prob_thresh={k: float(v) for k, v in merged["postprocess"]["prob_thresh"].items()},
        ),
    )
