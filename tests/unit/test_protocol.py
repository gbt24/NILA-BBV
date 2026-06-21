from bbv.verification.protocol import (
    build_protocol_summary,
    compute_false_claim_success_rate,
    decide_claim,
)


def test_decide_claim_accepts_only_threshold_and_margin() -> None:
    decision = decide_claim(
        owner_score=0.82,
        calibrated_threshold=0.7,
        competitor_scores={"owner1": 0.55, "owner2": 0.61},
        margin=0.1,
        query_budget_respected=True,
    )

    assert decision["decision"] is True
    assert decision["ambiguity_flag"] is False
    assert decision["nearest_competitor_owner_id"] == "owner2"
    assert abs(float(decision["nearest_competitor_margin"]) - 0.21) < 1e-8
    assert decision["rejection_reason"] is None


def test_decide_claim_rejects_competitor_inside_margin() -> None:
    decision = decide_claim(
        owner_score=0.82,
        calibrated_threshold=0.8,
        competitor_scores={"owner1": 0.77},
        margin=0.1,
        query_budget_respected=True,
    )

    assert decision["decision"] is False
    assert decision["ambiguity_flag"] is True
    assert decision["rejection_reason"] == "competitor_within_margin"


def test_compute_false_claim_success_rate_counts_passing_claims() -> None:
    rate = compute_false_claim_success_rate(
        false_claim_scores=[0.1, 0.71, 0.82, 0.3],
        calibrated_threshold=0.7,
    )

    assert rate == 0.5


def test_build_protocol_summary_uses_calibration_population() -> None:
    summary = build_protocol_summary(
        owner_id="owner0",
        owner_score=0.86,
        competitor_scores={"owner1": 0.4, "owner2": 0.5},
        calibration_owner_scores=[0.83, 0.86, 0.9],
        calibration_non_owner_scores=[0.2, 0.4, 0.55, 0.65],
        false_claim_scores=[0.3, 0.72],
        target_fpr=0.25,
        margin=0.1,
        query_budget=64,
        hard_label_only=True,
    )

    assert summary["owner_id"] == "owner0"
    assert summary["decision"] is True
    assert summary["calibration_sample_count"] == 7
    assert summary["owner_calibration_sample_count"] == 3
    assert summary["non_owner_calibration_sample_count"] == 4
    assert summary["false_claim_trial_count"] == 2
    assert summary["false_claim_success_rate"] >= 0.0
    assert summary["nearest_competitor_owner_id"] == "owner2"
    assert "calibration" in summary


def test_protocol_helpers_are_exported() -> None:
    from bbv.verification import build_protocol_summary as exported

    assert exported is build_protocol_summary
