from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClientStats:
    class_coverage: int
    skew_ratio: float
    main_wm_alignment: float
    privacy_penalty: float


def build_client_stats_from_histogram(
    histogram: dict[str, int], *, main_wm_alignment: float, privacy_penalty: float
) -> ClientStats:
    total = sum(histogram.values())
    if total == 0:
        skew_ratio = 1.0
    else:
        skew_ratio = max(histogram.values()) / total
    return ClientStats(
        class_coverage=len(histogram),
        skew_ratio=float(skew_ratio),
        main_wm_alignment=float(main_wm_alignment),
        privacy_penalty=float(privacy_penalty),
    )
