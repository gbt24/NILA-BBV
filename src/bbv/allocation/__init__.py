"""Adaptive allocation helpers for watermark budget assignment."""

from bbv.allocation.features import ClientStats, build_client_stats_from_histogram
from bbv.allocation.policy import allocate_watermark_budget, estimate_adaptability

__all__ = [
    "ClientStats",
    "build_client_stats_from_histogram",
    "allocate_watermark_budget",
    "estimate_adaptability",
]
