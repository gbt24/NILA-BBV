from pathlib import Path

from bbv.verification import (
    calibrate_threshold,
    compute_owner_score,
    recover_codeword,
    verify_owner,
)


def test_recover_codeword_maps_labels_to_bits() -> None:
    recovered = recover_codeword([0, 3, 2, 5])
    assert recovered == [0, 1, 0, 1]


def test_compute_owner_score_uses_negative_penalty() -> None:
    score = compute_owner_score(
        expected_codebook=[0, 1, 0, 1],
        recovered_codebook=[0, 1, 1, 1],
        negative_asr=0.5,
        negative_weight=0.2,
    )
    assert abs(score - 0.65) < 1e-8


def test_calibrate_threshold_separates_owner_and_non_owner_scores() -> None:
    calibration = calibrate_threshold(
        owner_scores=[0.82, 0.86, 0.88],
        non_owner_scores=[0.2, 0.35, 0.4],
        target_fpr=0.05,
    )
    assert 0.0 <= calibration["threshold"] <= 1.0
    assert calibration["owner_mean"] > calibration["non_owner_mean"]


def test_verify_owner_writes_calibration_artifact(tmp_path: Path) -> None:
    verification_path = tmp_path / "verification.json"
    calibration_path = tmp_path / "calibration.json"
    result = verify_owner(
        owner_id="owner0",
        expected_codebook=[0, 1, 0, 1],
        recovered_codebook=[0, 1, 0, 0],
        negative_asr=0.25,
        competitor_scores={"owner1": 0.3, "owner2": 0.4},
        threshold=0.5,
        margin=0.1,
        verification_path=verification_path,
        calibration_path=calibration_path,
    )
    assert result["owner_id"] == "owner0"
    assert "margin_value" in result
    assert calibration_path.exists()
