from __future__ import annotations


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_summary_metrics(
    *, main_rows: list[dict[str, object]], robustness_rows: list[dict[str, object]]
) -> dict[str, float]:
    total = len(main_rows)
    accepted = sum(int(bool(row.get("decision", False))) for row in main_rows)
    threshold_hits = sum(
        int(_safe_float(row.get("competitor_max", 0.0)) >= _safe_float(row.get("threshold", 0.5)))
        for row in main_rows
    )
    false_positive_like = sum(
        int(bool(row.get("decision", False)) and _safe_float(row.get("owner_score", 0.0)) < _safe_float(row.get("threshold", 0.5)))
        for row in main_rows
    )
    false_negative_like = sum(
        int((not bool(row.get("decision", False))) and _safe_float(row.get("owner_score", 0.0)) >= _safe_float(row.get("threshold", 0.5)))
        for row in main_rows
    )
    false_claim_acceptances = sum(
        int(bool(row.get("decision", False)) and _safe_float(row.get("competitor_max", 0.0)) >= _safe_float(row.get("threshold", 0.5)))
        for row in main_rows
    )

    denominator = total if total > 0 else 1
    return {
        "acceptance_rate": accepted / denominator,
        "ambiguity_rate": threshold_hits / denominator,
        "fpr": false_positive_like / denominator,
        "fnr": false_negative_like / denominator,
        "false_claim_acceptance_rate": false_claim_acceptances / denominator,
        "robustness_acceptance_rate": (
            sum(int(bool(row.get("decision", False))) for row in robustness_rows)
            / len(robustness_rows)
            if robustness_rows
            else 0.0
        ),
    }
