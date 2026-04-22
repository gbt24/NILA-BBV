from bbv.evaluation.stats import compute_summary_metrics


def test_compute_summary_metrics_contains_required_keys() -> None:
    metrics = compute_summary_metrics(
        main_rows=[
            {"decision": True, "threshold": 0.5, "owner_score": 0.7, "competitor_max": 0.3},
            {"decision": False, "threshold": 0.5, "owner_score": 0.4, "competitor_max": 0.2},
        ],
        robustness_rows=[{"decision": True}, {"decision": False}],
    )
    assert "acceptance_rate" in metrics
    assert "ambiguity_rate" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics


def test_compute_summary_metrics_includes_false_claim_rate() -> None:
    metrics = compute_summary_metrics(
        main_rows=[
            {"decision": True, "claim_type": "owner", "owner_score": 0.7, "threshold": 0.5},
            {"decision": True, "claim_type": "false_claim", "owner_score": 0.65, "threshold": 0.5},
        ],
        robustness_rows=[],
    )

    assert "false_claim_acceptance_rate" in metrics


def test_compute_summary_metrics_uses_false_claim_rows_for_false_claim_rate() -> None:
    metrics = compute_summary_metrics(
        main_rows=[
            {"decision": True, "claim_type": "owner", "owner_score": 0.7, "threshold": 0.5},
            {"decision": True, "claim_type": "false_claim", "owner_score": 0.65, "threshold": 0.5},
        ],
        robustness_rows=[],
    )

    assert metrics["false_claim_acceptance_rate"] == 1.0


def test_compute_summary_metrics_reports_zero_false_claim_rate_without_false_claim_rows() -> None:
    metrics = compute_summary_metrics(
        main_rows=[
            {"decision": True, "claim_type": "owner", "owner_score": 0.7, "threshold": 0.5},
            {"decision": False, "claim_type": "owner", "owner_score": 0.4, "threshold": 0.5},
        ],
        robustness_rows=[],
    )

    assert metrics["false_claim_acceptance_rate"] == 0.0
    assert metrics["fpr"] == 0.0


def test_compute_summary_metrics_does_not_use_false_claim_rows_for_owner_metrics() -> None:
    metrics = compute_summary_metrics(
        main_rows=[
            {"decision": True, "claim_type": "false_claim", "owner_score": 0.65, "threshold": 0.5},
            {"decision": False, "claim_type": "false_claim", "owner_score": 0.45, "threshold": 0.5},
        ],
        robustness_rows=[],
    )

    assert metrics["acceptance_rate"] == 0.0
    assert metrics["fnr"] == 0.0


def test_compute_summary_metrics_prefers_ambiguity_flag() -> None:
    metrics = compute_summary_metrics(
        main_rows=[
            {
                "decision": False,
                "threshold": 0.5,
                "owner_score": 0.9,
                "competitor_max": 0.1,
                "ambiguity_flag": True,
            }
        ],
        robustness_rows=[],
    )

    assert metrics["ambiguity_rate"] == 1.0
