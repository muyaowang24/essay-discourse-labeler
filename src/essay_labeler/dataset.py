from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from essay_labeler.labels import IGNORE_INDEX, NON_LABEL, label_to_id


@dataclass(slots=True)
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    word_ids: torch.Tensor
    labels: torch.Tensor | None = None


class EssayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length: int, labels: list[str], has_labels: bool):
        self.dataframe = list(dataframe)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id(labels)
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe[index]
        encoding = self.tokenizer(
            row["text"].split(),
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "word_ids": torch.tensor(
                [token_id if token_id is not None else NON_LABEL for token_id in word_ids],
                dtype=torch.long,
            ),
        }

        if self.has_labels:
            labels = []
            for word_id in word_ids:
                if word_id is None:
                    labels.append(IGNORE_INDEX)
                else:
                    labels.append(self.label_to_id[row["entities"][word_id]])
            item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item
