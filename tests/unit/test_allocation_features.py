from bbv.allocation.features import ClientStats
from bbv.allocation.policy import estimate_adaptability


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
