"""Minimal adaptive allocation policy for Phase 4."""

from __future__ import annotations

from math import exp


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def estimate_adaptability(
    *, client_label_histograms: list[dict[str, int]]
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for client_id, histogram in enumerate(client_label_histograms):
        total = sum(histogram.values())
        if total == 0:
            scores[client_id] = 0.0
            continue
        max_ratio = max(histogram.values()) / total
        class_coverage = len(histogram)
        raw_score = 1.5 * class_coverage - 4.0 * max_ratio
        scores[client_id] = _sigmoid(raw_score)
    return scores


def allocate_watermark_budget(
    *,
    adaptability_scores: dict[int, float],
    budget_clients: int,
    base_loss_weight: float,
) -> dict[int, dict[str, float | int | bool]]:
    if budget_clients <= 0:
        raise ValueError("budget_clients must be greater than 0")
    if base_loss_weight < 0.0:
        raise ValueError("base_loss_weight must be non-negative")

    ranked_client_ids = sorted(
        adaptability_scores, key=lambda client_id: adaptability_scores[client_id], reverse=True
    )
    selected = set(ranked_client_ids[: min(budget_clients, len(ranked_client_ids))])
    assignments: dict[int, dict[str, float | int | bool]] = {}
    for client_id, score in adaptability_scores.items():
        enabled = client_id in selected
        assignments[client_id] = {
            "enabled": enabled,
            "loss_weight": float(base_loss_weight * score) if enabled else 0.0,
            "depth": int(1 if enabled else 0),
            "score": float(score),
        }
    return assignments
