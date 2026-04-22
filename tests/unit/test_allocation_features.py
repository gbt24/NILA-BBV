from bbv.allocation.features import ClientStats
from bbv.allocation.policy import estimate_adaptability


def test_build_client_stats_from_gradients_reports_alignment() -> None:
    import torch

    from bbv.allocation.features import build_client_stats

    stats = build_client_stats(
        labels=torch.tensor([0, 0, 1, 1]),
        main_gradient=torch.tensor([1.0, 0.0]),
        wm_gradient=torch.tensor([0.5, 0.5]),
        privacy_penalty=0.1,
    )

    assert -1.0 <= stats.main_wm_alignment <= 1.0


def test_estimate_adaptability_uses_coverage_skew_and_alignment() -> None:
    stats = [
        ClientStats(
            class_coverage=8,
            skew_ratio=0.30,
            main_wm_alignment=0.8,
            privacy_penalty=0.1,
        ),
        ClientStats(
            class_coverage=2,
            skew_ratio=0.90,
            main_wm_alignment=-0.1,
            privacy_penalty=0.1,
        ),
    ]
    scores = estimate_adaptability(stats=stats)
    assert scores[0] > scores[1]
    assert 0.0 <= scores[0] <= 1.0


def test_estimate_adaptability_supports_histogram_callers() -> None:
    scores = estimate_adaptability(
        client_label_histograms=[{"0": 4, "1": 2}, {"0": 6}],
    )

    assert set(scores) == {0, 1}
