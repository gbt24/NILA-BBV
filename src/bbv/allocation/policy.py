"""Minimal adaptive allocation policy for Phase 4."""

from __future__ import annotations

from math import exp

from bbv.allocation.features import ClientStats


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def estimate_adaptability(
    *,
    client_label_histograms: list[dict[str, int]] | None = None,
    stats: list[ClientStats] | None = None,
) -> dict[int, float]:
    if stats is None:
        if client_label_histograms is None:
            raise ValueError("client_label_histograms or stats must be provided")
        stats = []
        for histogram in client_label_histograms:
            total = sum(histogram.values())
            skew_ratio = max(histogram.values()) / total if total > 0 else 1.0
            stats.append(
                ClientStats(
                    class_coverage=len(histogram),
                    skew_ratio=float(skew_ratio),
                    main_wm_alignment=0.0,
                    privacy_penalty=0.1,
                )
            )

    scores: dict[int, float] = {}
    for client_id, item in enumerate(stats):
        if item.class_coverage <= 0:
            scores[client_id] = 0.0
            continue
        raw_score = (
            0.8 * item.main_wm_alignment
            + 0.2 * item.class_coverage
            - 1.0 * item.skew_ratio
            - 0.5 * item.privacy_penalty
        )
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
