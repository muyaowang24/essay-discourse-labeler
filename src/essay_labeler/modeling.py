from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from essay_labeler.config import ExperimentConfig


class TokenClassificationModel(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, dropout_prob: float, multi_sample: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropout_prob + (idx * 0.1)) for idx in range(max(1, multi_sample))
        )
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = [self.classifier(dropout(hidden_states)) for dropout in self.dropouts]
        return torch.stack(logits).mean(dim=0)


def build_model(config: ExperimentConfig) -> TokenClassificationModel:
    return TokenClassificationModel(
        backbone_name=config.model.backbone_name,
        num_labels=config.num_labels,
        dropout_prob=config.model.dropout_prob,
        multi_sample=config.model.multi_sample_dropout,
    )


def build_tokenizer(config: ExperimentConfig):
    tokenizer_name = config.model.tokenizer_name or config.model.backbone_name
    return AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
