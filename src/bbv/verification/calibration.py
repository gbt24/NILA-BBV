"""Threshold calibration helpers for verification."""

from __future__ import annotations


def calibrate_threshold(
    *,
    owner_scores: list[float],
    non_owner_scores: list[float],
    target_fpr: float,
) -> dict[str, float | list[float]]:
    if not owner_scores or not non_owner_scores:
        raise ValueError("owner_scores and non_owner_scores must not be empty")
    if target_fpr < 0.0 or target_fpr >= 1.0:
        raise ValueError("target_fpr must be in [0, 1)")

    owner_scores = [float(score) for score in owner_scores]
    non_owner_scores = [float(score) for score in non_owner_scores]

    sorted_non_owner = sorted(non_owner_scores)
    raw_index = int((1.0 - target_fpr) * len(sorted_non_owner)) - 1
    index = min(max(raw_index, 0), len(sorted_non_owner) - 1)
    threshold = sorted_non_owner[index]

    return {
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "owner_mean": float(sum(owner_scores) / len(owner_scores)),
        "non_owner_mean": float(sum(non_owner_scores) / len(non_owner_scores)),
        "owner_scores": owner_scores,
        "non_owner_scores": non_owner_scores,
    }
