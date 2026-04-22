"""Query helpers for black-box verification."""

from __future__ import annotations

import torch

from bbv.watermarking.queries import adapt_queries_to_shape


def _prepare_queries(
    model: torch.nn.Module,
    queries: list[torch.Tensor],
    *,
    max_queries: int | None,
) -> list[torch.Tensor]:
    expected_shape = torch.Size(getattr(model, "_bbv_input_shape", (3, 32, 32)))
    adapted_queries = adapt_queries_to_shape(queries, expected_shape)
    if max_queries is None:
        return adapted_queries
    return adapted_queries[: max(0, int(max_queries))]


def batched_query_model(
    model: torch.nn.Module,
    queries: list[torch.Tensor],
    *,
    batch_size: int,
    max_queries: int | None = None,
) -> list[int]:
    prepared_queries = _prepare_queries(model, queries, max_queries=max_queries)
    predicted_labels: list[int] = []
    with torch.no_grad():
        for start in range(0, len(prepared_queries), batch_size):
            batch = torch.stack(prepared_queries[start : start + batch_size], dim=0)
            logits = model(batch)
            predicted_labels.extend(int(label) for label in logits.argmax(dim=1).tolist())
    return predicted_labels


def batched_query_model_logits(
    model: torch.nn.Module,
    queries: list[torch.Tensor],
    *,
    batch_size: int,
    max_queries: int | None = None,
) -> list[torch.Tensor]:
    prepared_queries = _prepare_queries(model, queries, max_queries=max_queries)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prepared_queries), batch_size):
            batch = torch.stack(prepared_queries[start : start + batch_size], dim=0)
            logits = model(batch)
            outputs.extend(row.detach().cpu() for row in logits)
    return outputs


def query_model(model: torch.nn.Module, queries: list[torch.Tensor]) -> list[int]:
    return batched_query_model(model, queries, batch_size=1)


def query_model_logits(model: torch.nn.Module, queries: list[torch.Tensor]) -> list[torch.Tensor]:
    return batched_query_model_logits(model, queries, batch_size=1)
