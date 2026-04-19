from bbv.allocation import allocate_watermark_budget, estimate_adaptability


def test_estimate_adaptability_scores_in_range() -> None:
    scores = estimate_adaptability(
        client_label_histograms=[
            {"0": 8, "1": 8},
            {"0": 15, "1": 1},
            {"0": 3, "1": 3, "2": 2},
        ]
    )

    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_allocate_watermark_budget_conserves_budget_count() -> None:
    scores = {0: 0.1, 1: 0.6, 2: 0.9, 3: 0.2}
    assignments = allocate_watermark_budget(
        adaptability_scores=scores,
        budget_clients=2,
        base_loss_weight=0.2,
    )

    enabled = [client_id for client_id, item in assignments.items() if item["enabled"]]
    assert len(enabled) == 2
    assert set(enabled) == {1, 2}
    assert all(0.0 <= item["loss_weight"] for item in assignments.values())
