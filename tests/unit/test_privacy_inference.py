from bbv.privacy import infer_label_skew_from_client_stats


def test_label_distribution_attack_returns_auc() -> None:
    result = infer_label_skew_from_client_stats(
        client_stats=[
            {"class_coverage": 2, "skew_ratio": 0.9, "main_wm_alignment": 0.8, "privacy_penalty": 0.1},
            {"class_coverage": 8, "skew_ratio": 0.2, "main_wm_alignment": 0.1, "privacy_penalty": 0.1},
        ],
        skew_targets=[1, 0],
    )
    assert 0.0 <= result["auc"] <= 1.0
