from __future__ import annotations


SUMMARY_METRIC_KEYS = (
    "acceptance_rate",
    "ambiguity_rate",
    "fpr",
    "fnr",
    "false_claim_acceptance_rate",
    "robustness_acceptance_rate",
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


def compute_summary_metrics(
    *, main_rows: list[dict[str, object]], robustness_rows: list[dict[str, object]]
) -> dict[str, float]:
    owner_rows = _rows_for_claim_type(main_rows, "owner")
    false_claim_rows = _rows_for_claim_type(main_rows, "false_claim")

    return {
        "acceptance_rate": _rate(
            owner_rows,
            lambda row: bool(row.get("decision", False)),
        ),
        "ambiguity_rate": _rate(
            owner_rows,
            lambda row: bool(row.get("ambiguity_flag", False))
            if "ambiguity_flag" in row
            else _safe_float(row.get("competitor_max", 0.0))
            >= _safe_float(row.get("threshold", 0.5)),
        ),
        "fpr": _rate(
            false_claim_rows,
            lambda row: bool(row.get("decision", False)),
        ),
        "fnr": _rate(
            owner_rows,
            lambda row: not bool(row.get("decision", False)),
        ),
        "false_claim_acceptance_rate": _rate(
            false_claim_rows,
            lambda row: bool(row.get("decision", False)),
        ),
        "robustness_acceptance_rate": (
            sum(int(bool(row.get("decision", False))) for row in robustness_rows)
            / len(robustness_rows)
            if robustness_rows
            else 0.0
        ),
    }
