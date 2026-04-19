from bbv.verification import calibrate_threshold


def test_calibrate_threshold_respects_target_fpr() -> None:
    calibration = calibrate_threshold(
        owner_scores=[0.88, 0.84, 0.90],
        non_owner_scores=[0.10, 0.22, 0.31, 0.45],
        target_fpr=0.25,
    )
    assert calibration["threshold"] >= 0.31
    assert calibration["target_fpr"] == 0.25
    assert "owner_scores" in calibration
    assert "non_owner_scores" in calibration
