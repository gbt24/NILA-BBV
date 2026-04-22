from __future__ import annotations

from bbv.evaluation.significance import exact_binomial_ci


SUMMARY_METRIC_KEYS = (
    "acceptance_rate",
    "ambiguity_rate",
    "fpr",
    "fnr",
    "false_claim_acceptance_rate",
    "robustness_acceptance_rate",
    "privacy_leakage_auc",
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rows_for_claim_type(
    rows: list[dict[str, object]],
    claim_type: str,
) -> list[dict[str, object]]:
    if claim_type == "owner":
        return [
            row
            for row in rows
            if str(row.get("claim_type", "owner")).lower() == "owner"
        ]
    return [
        row
        for row in rows
        if str(row.get("claim_type", "")).lower() == claim_type
    ]


def _rate(rows: list[dict[str, object]], predicate: object) -> float:
    if not rows:
        return 0.0
    return sum(int(predicate(row)) for row in rows) / len(rows)


def _rate_with_ci(rows: list[dict[str, object]], predicate: object) -> tuple[float, tuple[float, float]]:
    if not rows:
        return (0.0, (0.0, 0.0))
    successes = sum(int(predicate(row)) for row in rows)
    rate = successes / len(rows)
    return (rate, exact_binomial_ci(successes=successes, total=len(rows), alpha=0.05))


def compute_summary_metrics(
    *, main_rows: list[dict[str, object]], robustness_rows: list[dict[str, object]]
) -> dict[str, object]:
    owner_rows = _rows_for_claim_type(main_rows, "owner")
    false_claim_rows = _rows_for_claim_type(main_rows, "false_claim")

    acceptance_rate, acceptance_rate_ci = _rate_with_ci(
        owner_rows,
        lambda row: bool(row.get("decision", False)),
    )
    ambiguity_rate, ambiguity_rate_ci = _rate_with_ci(
        owner_rows,
        lambda row: bool(row.get("ambiguity_flag", False))
        if "ambiguity_flag" in row
        else _safe_float(row.get("competitor_max", 0.0)) >= _safe_float(row.get("threshold", 0.5)),
    )
    fpr, fpr_ci = _rate_with_ci(false_claim_rows, lambda row: bool(row.get("decision", False)))
    fnr, fnr_ci = _rate_with_ci(owner_rows, lambda row: not bool(row.get("decision", False)))
    false_claim_acceptance_rate, false_claim_acceptance_rate_ci = _rate_with_ci(
        false_claim_rows,
        lambda row: bool(row.get("decision", False)),
    )
    robustness_acceptance_rate, robustness_acceptance_rate_ci = _rate_with_ci(
        robustness_rows,
        lambda row: bool(row.get("decision", False)),
    )

    return {
        "acceptance_rate": acceptance_rate,
        "acceptance_rate_ci": acceptance_rate_ci,
        "ambiguity_rate": ambiguity_rate,
        "ambiguity_rate_ci": ambiguity_rate_ci,
        "fpr": fpr,
        "fpr_ci": fpr_ci,
        "fnr": fnr,
        "fnr_ci": fnr_ci,
        "false_claim_acceptance_rate": false_claim_acceptance_rate,
        "false_claim_acceptance_rate_ci": false_claim_acceptance_rate_ci,
        "robustness_acceptance_rate": robustness_acceptance_rate,
        "robustness_acceptance_rate_ci": robustness_acceptance_rate_ci,
    }
