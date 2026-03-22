from __future__ import annotations

DEFAULT_LABELS = [
    "O",
    "B-Lead",
    "I-Lead",
    "B-Position",
    "I-Position",
    "B-Claim",
    "I-Claim",
    "B-Counterclaim",
    "I-Counterclaim",
    "B-Rebuttal",
    "I-Rebuttal",
    "B-Evidence",
    "I-Evidence",
    "B-Concluding Statement",
    "I-Concluding Statement",
]

IGNORE_INDEX = -100
NON_LABEL = -1


def label_to_id(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def id_to_label(labels: list[str]) -> dict[int, str]:
    return {idx: label for idx, label in enumerate(labels)}


def discourse_to_bio(discourse_type: str, token_ids: list[int], total_tokens: int) -> list[str]:
    entities = ["O"] * total_tokens
    if not token_ids:
        return entities
    entities[token_ids[0]] = f"B-{discourse_type}"
    for token_id in token_ids[1:]:
        entities[token_id] = f"I-{discourse_type}"
    return entities

