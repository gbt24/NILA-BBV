from __future__ import annotations

import torch
import torch.nn as nn

from bbv.datasets.text import TEXT_SEQUENCE_LENGTH, TEXT_VOCAB_SIZE


class TextCNN(nn.Module):
    def __init__(self, *, num_classes: int, sequence_length: int = TEXT_SEQUENCE_LENGTH) -> None:
        super().__init__()
        self.embedding = nn.Embedding(TEXT_VOCAB_SIZE, 64, padding_idx=0)
        self.conv = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(64, num_classes)
        self.sequence_length = sequence_length

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        channels_first = embedded.transpose(1, 2)
        features = self.activation(self.conv(channels_first))
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled)


def build_text_cnn(num_classes: int, input_shape: tuple[int, ...] | None = None) -> nn.Module:
    if input_shape is None or len(input_shape) != 1:
        sequence_length = TEXT_SEQUENCE_LENGTH
    else:
        sequence_length = int(input_shape[0])
    return TextCNN(num_classes=num_classes, sequence_length=sequence_length)
