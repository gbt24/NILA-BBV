"""Threshold calibration helpers for verification."""

from __future__ import annotations


def _compute_auc(owner_scores: list[float], non_owner_scores: list[float]) -> float:
    wins = 0.0
    total = len(owner_scores) * len(non_owner_scores)
    if total == 0:
        return 0.0
    for owner_score in owner_scores:
        for non_owner_score in non_owner_scores:
            if owner_score > non_owner_score:
                wins += 1.0
            elif owner_score == non_owner_score:
                wins += 0.5
    return wins / total


def _build_roc_points(owner_scores: list[float], non_owner_scores: list[float]) -> list[dict[str, float]]:
    thresholds = sorted(set(owner_scores + non_owner_scores), reverse=True)
    thresholds = [float("inf")] + thresholds + [float("-inf")]
    points: list[dict[str, float]] = []
    for threshold in thresholds:
        tpr = sum(int(score >= threshold) for score in owner_scores) / len(owner_scores)
        fpr = sum(int(score >= threshold) for score in non_owner_scores) / len(non_owner_scores)
        points.append({"threshold": float(threshold), "tpr": float(tpr), "fpr": float(fpr)})
    return points


def _threshold_interval(sorted_non_owner: list[float], index: int) -> list[float]:
    lower = sorted_non_owner[max(index - 1, 0)]
    upper = sorted_non_owner[min(index + 1, len(sorted_non_owner) - 1)]
    return [float(lower), float(upper)]


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
    achieved_fpr = sum(int(score >= threshold) for score in non_owner_scores) / len(non_owner_scores)
    achieved_tpr = sum(int(score >= threshold) for score in owner_scores) / len(owner_scores)
    roc_points = _build_roc_points(owner_scores, non_owner_scores)
    auc = _compute_auc(owner_scores, non_owner_scores)

    return {
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(achieved_fpr),
        "achieved_tpr": float(achieved_tpr),
        "auc": float(auc),
        "roc_points": roc_points,
        "threshold_ci": _threshold_interval(sorted_non_owner, index),
        "owner_mean": float(sum(owner_scores) / len(owner_scores)),
        "non_owner_mean": float(sum(non_owner_scores) / len(non_owner_scores)),
        "owner_scores": owner_scores,
        "non_owner_scores": non_owner_scores,
    }
