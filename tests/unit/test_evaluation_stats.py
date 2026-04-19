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
