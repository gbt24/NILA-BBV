"""Protocol-level decision helpers for anti-false-claim verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bbv.verification.calibration import calibrate_threshold


def _as_float_list(scores: Sequence[float], *, name: str) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _nearest_competitor(
    competitor_scores: Mapping[str, float],
) -> tuple[str | None, float | None]:
    if not competitor_scores:
        return None, None
    owner_id, score = max(
        ((str(owner_id), float(score)) for owner_id, score in competitor_scores.items()),
        key=lambda item: item[1],
    )
    return owner_id, score


def decide_claim(
    *,
    owner_score: float,
    calibrated_threshold: float,
    competitor_scores: Mapping[str, float],
    margin: float,
    query_budget_respected: bool,
) -> dict[str, object]:
    """Return the calibrated accept/reject decision for one ownership claim."""

    owner_score = float(owner_score)
    calibrated_threshold = float(calibrated_threshold)
    margin = float(margin)
    nearest_id, nearest_score = _nearest_competitor(competitor_scores)
    nearest_margin = None if nearest_score is None else owner_score - nearest_score
    threshold_passed = owner_score >= calibrated_threshold
    competitor_passed = nearest_score is not None and nearest_score >= calibrated_threshold
    competitor_within_margin = nearest_margin is not None and nearest_margin < margin
    ambiguity_flag = bool(competitor_passed or competitor_within_margin)

    rejection_reason: str | None = None
    if not query_budget_respected:
        rejection_reason = "query_budget_exhausted"
    elif not threshold_passed:
        rejection_reason = "below_calibrated_threshold"
    elif competitor_passed:
        rejection_reason = "competitor_passed_threshold"
    elif competitor_within_margin:
        rejection_reason = "competitor_within_margin"

    accepted = rejection_reason is None
    return {
        "decision": accepted,
        "owner_score": owner_score,
        "calibrated_threshold": calibrated_threshold,
        "nearest_competitor_owner_id": nearest_id,
        "nearest_competitor_score": nearest_score,
        "nearest_competitor_margin": nearest_margin,
        "ambiguity_flag": ambiguity_flag,
        "margin": margin,
        "query_budget_respected": bool(query_budget_respected),
        "rejection_reason": rejection_reason,
    }


def compute_false_claim_success_rate(
    *,
    false_claim_scores: Sequence[float],
    calibrated_threshold: float,
) -> float:
    scores = [float(score) for score in false_claim_scores]
    if not scores:
        return 0.0
    threshold = float(calibrated_threshold)
    return sum(int(score >= threshold) for score in scores) / len(scores)


def build_protocol_summary(
    *,
    owner_id: str,
    owner_score: float,
    competitor_scores: Mapping[str, float],
    calibration_owner_scores: Sequence[float],
    calibration_non_owner_scores: Sequence[float],
    false_claim_scores: Sequence[float],
    target_fpr: float,
    margin: float,
    query_budget: int | None,
    hard_label_only: bool,
    smoke_only: bool = False,
) -> dict[str, Any]:
    owner_population = _as_float_list(calibration_owner_scores, name="calibration_owner_scores")
    non_owner_population = _as_float_list(
        calibration_non_owner_scores,
        name="calibration_non_owner_scores",
    )
    calibration = calibrate_threshold(
        owner_scores=owner_population,
        non_owner_scores=non_owner_population,
        target_fpr=float(target_fpr),
    )
    threshold = float(calibration["threshold"])
    decision = decide_claim(
        owner_score=float(owner_score),
        calibrated_threshold=threshold,
        competitor_scores=competitor_scores,
        margin=float(margin),
        query_budget_respected=True,
    )
    false_claim_scores_list = [float(score) for score in false_claim_scores]
    false_claim_success_rate = compute_false_claim_success_rate(
        false_claim_scores=false_claim_scores_list,
        calibrated_threshold=threshold,
    )

    return {
        "owner_id": str(owner_id),
        **decision,
        "competitor_scores": {str(k): float(v) for k, v in competitor_scores.items()},
        "false_claim_scores": false_claim_scores_list,
        "false_claim_success_rate": false_claim_success_rate,
        "false_claim_trial_count": len(false_claim_scores_list),
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(calibration["achieved_fpr"]),
        "achieved_tpr": float(calibration["achieved_tpr"]),
        "auc": float(calibration["auc"]),
        "calibration": calibration,
        "calibration_sample_count": len(owner_population) + len(non_owner_population),
        "owner_calibration_sample_count": len(owner_population),
        "non_owner_calibration_sample_count": len(non_owner_population),
        "query_budget": query_budget,
        "hard_label_only": bool(hard_label_only),
        "smoke_only": bool(smoke_only),
    }
