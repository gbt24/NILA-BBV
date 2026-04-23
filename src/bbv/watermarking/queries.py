from __future__ import annotations

import torch
import torch.nn.functional as F


def _build_single_query(seed: int, bit: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(3, 32, 32, generator=generator)
    if bit == 1:
        query = query + 0.2
    return query


def build_positive_queries(codebook: list[int], seed: int) -> list[torch.Tensor]:
    return [
        _build_single_query(seed=seed + 97 * index, bit=bit)
        for index, bit in enumerate(codebook)
    ]


def build_negative_queries(codebook: list[int], seed: int) -> list[torch.Tensor]:
    flipped = [1 - bit for bit in codebook]
    return build_positive_queries(codebook=flipped, seed=seed + 104729)


def adapt_queries_to_shape(
    queries: list[torch.Tensor], feature_shape: torch.Size
) -> list[torch.Tensor]:
    if len(feature_shape) == 1:
        sequence_length = int(feature_shape[0])
        adapted_tokens: list[torch.Tensor] = []
        for query in queries:
            flattened = query.flatten()
            if flattened.numel() < sequence_length:
                repeats = (sequence_length + flattened.numel() - 1) // flattened.numel()
                flattened = flattened.repeat(repeats)
            trimmed = flattened[:sequence_length]
            normalized = torch.sigmoid(trimmed)
            token_ids = torch.clamp((normalized * 2047).round(), min=0, max=2047).to(torch.long)
            adapted_tokens.append(token_ids)
        return adapted_tokens

    channels, height, width = feature_shape
    adapted: list[torch.Tensor] = []
    for query in queries:
        adjusted = query
        if adjusted.dim() == 2:
            adjusted = adjusted.unsqueeze(0)
        if adjusted.shape[0] != channels:
            adjusted = adjusted.mean(dim=0, keepdim=True).repeat(channels, 1, 1)
        if adjusted.shape[-2:] != (height, width):
            adjusted = F.interpolate(
                adjusted.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        adapted.append(adjusted)
    return adapted
