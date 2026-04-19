"""Summarize standard output files into report-ready tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationSummary:
    main_rows: list[dict[str, object]]
    ablation_rows: list[dict[str, object]]
    robustness_rows: list[dict[str, object]]
    metrics: dict[str, float]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_outputs(results_root: Path, attacks_root: Path | None = None) -> EvaluationSummary:
    results_root = Path(results_root)
    main_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []

    if results_root.exists():
        run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
        if (results_root / "verification_margin_summary.json").exists():
            run_dirs = [results_root] + run_dirs
        for run_dir in run_dirs:
            summary_path = run_dir / "verification_margin_summary.json"
            if not summary_path.exists():
                continue
            verification = _load_json(summary_path)
            metadata = _load_json(run_dir / "run_metadata.json") if (run_dir / "run_metadata.json").exists() else {}
            competitor_scores = verification.get("competitor_scores", {})
            competitor_max = 0.0
            if isinstance(competitor_scores, dict) and competitor_scores:
                competitor_max = max(_safe_float(v) for v in competitor_scores.values())

            row = {
                "run_id": run_dir.name,
                "owner_id": str(verification.get("owner_id", "unknown")),
                "owner_score": _safe_float(verification.get("owner_score")),
                "decision": bool(verification.get("decision", False)),
                "threshold": _safe_float(verification.get("threshold"), 0.5),
                "margin_value": _safe_float(verification.get("margin_value")),
                "competitor_max": competitor_max,
            }
            main_rows.append(row)
            ablation_rows.append(
                {
                    "run_id": run_dir.name,
                    "allocation_enabled": bool(
                        (metadata.get("allocation") or {}).get("enabled", False)
                        if isinstance(metadata.get("allocation"), dict)
                        else False
                    ),
                    "owner_score": row["owner_score"],
                    "decision": row["decision"],
                }
            )

    robustness_rows: list[dict[str, object]] = []
    if attacks_root is not None:
        attacks_root = Path(attacks_root)
        if attacks_root.exists():
            attack_dirs = sorted(path for path in attacks_root.iterdir() if path.is_dir())
            for attack_dir in attack_dirs:
                attack_log_path = attack_dir / "attack_log.json"
                verification_path = attack_dir / "verification_after_attack.json"
                if not attack_log_path.exists() or not verification_path.exists():
                    continue
                attack_log = _load_json(attack_log_path)
                verification = _load_json(verification_path)
                robustness_rows.append(
                    {
                        "attack_run": attack_dir.name,
                        "attack": str(attack_log.get("attack", "unknown")),
                        "owner_score": _safe_float(verification.get("owner_score")),
                        "decision": bool(verification.get("decision", False)),
                    }
                )

    total = len(main_rows)
    accepted = sum(int(bool(row["decision"])) for row in main_rows)
    threshold_hits = sum(
        int(_safe_float(row["competitor_max"]) >= _safe_float(row["threshold"]))
        for row in main_rows
    )
    false_positive_like = sum(
        int(bool(row["decision"]) and _safe_float(row["owner_score"]) < _safe_float(row["threshold"]))
        for row in main_rows
    )
    false_negative_like = sum(
        int((not bool(row["decision"])) and _safe_float(row["owner_score"]) >= _safe_float(row["threshold"]))
        for row in main_rows
    )
    false_claim_acceptances = sum(
        int(bool(row["decision"]) and _safe_float(row["competitor_max"]) >= _safe_float(row["threshold"]))
        for row in main_rows
    )

    denominator = total if total > 0 else 1
    metrics = {
        "acceptance_rate": accepted / denominator,
        "ambiguity_rate": threshold_hits / denominator,
        "fpr": false_positive_like / denominator,
        "fnr": false_negative_like / denominator,
        "false_claim_acceptance_rate": false_claim_acceptances / denominator,
        "robustness_acceptance_rate": (
            sum(int(bool(row["decision"])) for row in robustness_rows) / len(robustness_rows)
            if robustness_rows
            else 0.0
        ),
    }

    return EvaluationSummary(
        main_rows=main_rows,
        ablation_rows=ablation_rows,
        robustness_rows=robustness_rows,
        metrics=metrics,
    )
