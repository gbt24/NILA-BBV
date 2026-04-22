from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClientStats:
    class_coverage: int
    skew_ratio: float
    main_wm_alignment: float
    privacy_penalty: float


def _compute_skew_ratio(histogram: dict[str, int]) -> float:
    total = sum(histogram.values())
    if total == 0:
        return 1.0
    return float(max(histogram.values()) / total)


def _build_histogram(labels: torch.Tensor) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for label in labels.tolist():
        key = str(int(label))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.norm(left)
    right_norm = torch.linalg.norm(right)
    if left_norm.item() == 0.0 or right_norm.item() == 0.0:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm.item() * right_norm.item()))


def build_client_stats(
    *,
    labels: torch.Tensor,
    main_gradient: torch.Tensor,
    wm_gradient: torch.Tensor,
    privacy_penalty: float,
) -> ClientStats:
    histogram = _build_histogram(labels)
    return ClientStats(
        class_coverage=len(histogram),
        skew_ratio=_compute_skew_ratio(histogram),
        main_wm_alignment=_cosine_similarity(main_gradient, wm_gradient),
        privacy_penalty=float(privacy_penalty),
    )


def build_client_stats_from_histogram(
    histogram: dict[str, int], *, main_wm_alignment: float, privacy_penalty: float
) -> ClientStats:
    return ClientStats(
        class_coverage=len(histogram),
        skew_ratio=_compute_skew_ratio(histogram),
        main_wm_alignment=float(main_wm_alignment),
        privacy_penalty=float(privacy_penalty),
    )
