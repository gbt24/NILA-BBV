from pathlib import Path

import torch
import pytest

from bbv.verification import compute_owner_score
from bbv.verification.query import batched_query_model


def test_compute_owner_score_matches_hamming_formula() -> None:
    score = compute_owner_score(
        expected_codebook=[0, 1, 0, 1],
        recovered_codebook=[0, 1, 1, 1],
        negative_asr=0.25,
        negative_weight=0.2,
    )
    assert score == 1.0 - (1 / 4) - 0.2 * 0.25


def test_query_model_supports_hard_label_budget() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    queries = [torch.zeros(3, 32, 32) for _ in range(80)]

    outputs = batched_query_model(model, queries, batch_size=16, max_queries=64)

    assert len(outputs) <= 64


def test_run_verification_rejects_zero_query_budget(tmp_path) -> None:
    from bbv.verification import run_verification_from_checkpoint
    from bbv.watermarking import save_owner_artifacts

    from bbv.models import build_model

    checkpoint_path = tmp_path / "checkpoint.pt"
    model = build_model("mlp", num_classes=10, input_shape=(3, 32, 32))
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "mlp",
            "num_classes": 10,
            "input_shape": (3, 32, 32),
        },
        checkpoint_path,
    )
    artifacts_path = tmp_path / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts_path,
        owner_id="owner0",
        codebook=[0, 1],
        queries=[torch.zeros(3, 32, 32), torch.zeros(3, 32, 32)],
        negative_queries=[torch.zeros(3, 32, 32), torch.zeros(3, 32, 32)],
    )

    with pytest.raises(ValueError, match="query_budget"):
        run_verification_from_checkpoint(
            checkpoint_path=checkpoint_path,
            artifacts_path=artifacts_path,
            verification_path=tmp_path / "verification.json",
            calibration_path=tmp_path / "calibration.json",
            decision_threshold=0.5,
            margin=0.05,
            competitor_owner_ids=["owner1"],
            seed=0,
            query_budget=0,
        )


def test_compute_negative_asr_counts_matches_against_flipped_codebook() -> None:
    from bbv.verification.baseline import compute_negative_asr

    negative_codebook = [1, 0, 1, 0]
    neg_recovered = [1, 1, 1, 0]

    assert compute_negative_asr(negative_codebook, neg_recovered) == 3 / 4


def test_verify_owner_marks_ambiguity_when_competitor_is_within_margin() -> None:
    from bbv.verification.baseline import verify_owner

    summary = verify_owner(
        owner_id="owner0",
        expected_codebook=[0, 1, 0, 1],
        recovered_codebook=[0, 1, 0, 1],
        negative_asr=0.0,
        competitor_scores={"owner1": 0.97},
        threshold=0.99,
        margin=0.05,
        verification_path=Path("/tmp/verification-owner.json"),
        calibration_path=Path("/tmp/calibration-owner.json"),
    )

    assert summary["decision"] is False
    assert summary["ambiguity_flag"] is True


def test_verify_owner_does_not_mark_low_score_case_as_ambiguous() -> None:
    from bbv.verification.baseline import verify_owner

    summary = verify_owner(
        owner_id="owner0",
        expected_codebook=[0, 1, 0, 1],
        recovered_codebook=[0, 1, 1, 0],
        negative_asr=0.3,
        competitor_scores={"owner1": 0.18},
        threshold=0.5,
        margin=0.05,
        verification_path=Path("/tmp/verification-owner-low.json"),
        calibration_path=Path("/tmp/calibration-owner-low.json"),
    )

    assert summary["owner_score"] < 0.5
    assert summary["ambiguity_flag"] is False
