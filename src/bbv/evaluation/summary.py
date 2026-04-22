"""Summarize standard output files into report-ready tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bbv.evaluation.stats import compute_summary_metrics
from bbv.privacy import infer_label_skew_from_client_stats


MAIN_RESULT_COLUMNS = (
    "run_id",
    "owner_id",
    "claim_type",
    "owner_score",
    "decision",
    "threshold",
    "margin_value",
    "competitor_max",
    "ambiguity_flag",
)

ABLATION_RESULT_COLUMNS = (
    "run_id",
    "allocation_enabled",
    "owner_score",
    "decision",
)

ROBUSTNESS_RESULT_COLUMNS = (
    "attack_run",
    "attack",
    "owner_score",
    "decision",
)


@dataclass(frozen=True)
class EvaluationSummary:
    main_rows: list[dict[str, object]]
    ablation_rows: list[dict[str, object]]
    robustness_rows: list[dict[str, object]]
    metrics: dict[str, object]
    privacy_leakage_auc: float = 0.5
    hypothesis_verdicts: dict[str, dict[str, str]] | None = None


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_columns(row: dict[str, object], columns: tuple[str, ...]) -> dict[str, object]:
    return {column: row.get(column, "") for column in columns}


def _build_hypothesis_verdicts(metrics: dict[str, object]) -> dict[str, dict[str, str]]:
    acceptance_rate = _safe_float(metrics.get("acceptance_rate"))
    ambiguity_rate = _safe_float(metrics.get("ambiguity_rate"), 1.0)
    fpr = _safe_float(metrics.get("fpr"), 1.0)
    robustness_acceptance_rate = _safe_float(metrics.get("robustness_acceptance_rate"))
    privacy_leakage_auc = _safe_float(metrics.get("privacy_leakage_auc"), 0.5)

    def verdict(condition_supported: bool, condition_mixed: bool, evidence: str) -> dict[str, str]:
        if condition_supported:
            label = "supported"
        elif condition_mixed:
            label = "mixed"
        else:
            label = "unsupported"
        return {"label": label, "evidence": evidence}

    return {
        "H1": verdict(
            acceptance_rate >= 0.8,
            acceptance_rate >= 0.5,
            f"acceptance_rate={acceptance_rate:.3f}",
        ),
        "H2": verdict(
            ambiguity_rate <= 0.1 and fpr <= 0.1,
            ambiguity_rate <= 0.25 and fpr <= 0.25,
            f"ambiguity_rate={ambiguity_rate:.3f}, fpr={fpr:.3f}",
        ),
        "H3": verdict(
            acceptance_rate >= 0.8,
            acceptance_rate >= 0.5,
            f"acceptance_rate={acceptance_rate:.3f}",
        ),
        "H4": verdict(
            robustness_acceptance_rate >= 0.7,
            robustness_acceptance_rate >= 0.4,
            f"robustness_acceptance_rate={robustness_acceptance_rate:.3f}",
        ),
        "H5": verdict(
            privacy_leakage_auc <= 0.6,
            privacy_leakage_auc <= 0.75,
            f"privacy_leakage_auc={privacy_leakage_auc:.3f}",
        ),
    }


def summarize_outputs(results_root: Path, attacks_root: Path | None = None) -> EvaluationSummary:
    results_root = Path(results_root)
    main_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []

    if results_root.exists():
        run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
        if (results_root / "verification_margin_summary.json").exists():
            run_dirs = [results_root] + run_dirs
        privacy_stats: list[dict[str, float | int]] = []
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

            row = _select_columns(
                {
                "run_id": run_dir.name,
                "owner_id": str(verification.get("owner_id", "unknown")),
                "claim_type": str(verification.get("claim_type", "owner")),
                "owner_score": _safe_float(verification.get("owner_score")),
                "decision": bool(verification.get("decision", False)),
                "threshold": _safe_float(verification.get("threshold"), 0.5),
                "margin_value": _safe_float(verification.get("margin_value")),
                "competitor_max": competitor_max,
                "ambiguity_flag": bool(verification.get("ambiguity_flag", False)),
                },
                MAIN_RESULT_COLUMNS,
            )
            main_rows.append(row)
            ablation_rows.append(
                _select_columns(
                    {
                        "run_id": run_dir.name,
                        "allocation_enabled": bool(
                            (metadata.get("allocation") or {}).get("enabled", False)
                            if isinstance(metadata.get("allocation"), dict)
                            else False
                        ),
                        "owner_score": row["owner_score"],
                        "decision": row["decision"],
                    },
                    ABLATION_RESULT_COLUMNS,
                )
            )

            allocation_path = run_dir / "allocation_assignments.json"
            if allocation_path.exists():
                allocation_payload = _load_json(allocation_path)
                round_assignments = allocation_payload.get("round_assignments", [])
                if isinstance(round_assignments, list):
                    for round_assignment in round_assignments:
                        assignments = round_assignment.get("assignments", {})
                        if not isinstance(assignments, dict):
                            continue
                        for assignment in assignments.values():
                            if not isinstance(assignment, dict):
                                continue
                            stats = assignment.get("stats")
                            if isinstance(stats, dict):
                                privacy_stats.append(stats)

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
                    _select_columns(
                        {
                            "attack_run": attack_dir.name,
                            "attack": str(attack_log.get("attack", "unknown")),
                            "owner_score": _safe_float(verification.get("owner_score")),
                            "decision": bool(verification.get("decision", False)),
                        },
                        ROBUSTNESS_RESULT_COLUMNS,
                    )
                )

    metrics = compute_summary_metrics(
        main_rows=main_rows,
        robustness_rows=robustness_rows,
    )
    privacy_leakage_auc = 0.5
    if privacy_stats:
        skew_targets = [int(float(item.get("skew_ratio", 0.0)) >= 0.5) for item in privacy_stats]
        inference = infer_label_skew_from_client_stats(
            client_stats=privacy_stats,
            skew_targets=skew_targets,
        )
        privacy_leakage_auc = float(inference["auc"])
        metrics["privacy_leakage_auc"] = privacy_leakage_auc
    else:
        metrics["privacy_leakage_auc"] = privacy_leakage_auc
    hypothesis_verdicts = _build_hypothesis_verdicts(metrics)

    return EvaluationSummary(
        main_rows=main_rows,
        ablation_rows=ablation_rows,
        robustness_rows=robustness_rows,
        metrics=metrics,
        privacy_leakage_auc=privacy_leakage_auc,
        hypothesis_verdicts=hypothesis_verdicts,
    )
