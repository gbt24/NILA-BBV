"""Black-box verification with score, margin, and calibration."""

from __future__ import annotations

from pathlib import Path

import torch

from bbv.models import build_model
from bbv.utils.io import write_json
from bbv.watermarking import generate_codebook, load_owner_artifacts


def _load_model_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    num_classes = int(state_dict["fc.weight"].shape[0])
    model = build_model("resnet18", num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def query_model(model: torch.nn.Module, queries: list[torch.Tensor]) -> list[int]:
    predicted_labels: list[int] = []
    with torch.no_grad():
        for query in queries:
            logits = model(query.unsqueeze(0))
            predicted_labels.append(int(logits.argmax(dim=1).item()))
    return predicted_labels


def recover_codeword(predicted_labels: list[int]) -> list[int]:
    return [int(label % 2) for label in predicted_labels]


def compute_owner_score(
    *,
    expected_codebook: list[int],
    recovered_codebook: list[int],
    negative_asr: float,
    negative_weight: float,
) -> float:
    hamming_distance = sum(
        int(a != b) for a, b in zip(expected_codebook, recovered_codebook)
    )
    score = 1.0 - (hamming_distance / len(expected_codebook)) - negative_weight * negative_asr
    return float(score)


def calibrate_threshold(
    *,
    owner_scores: list[float],
    non_owner_scores: list[float],
    target_fpr: float,
) -> dict[str, float | list[float]]:
    if not owner_scores or not non_owner_scores:
        raise ValueError("owner_scores and non_owner_scores must not be empty")
    if target_fpr < 0.0 or target_fpr >= 1.0:
        raise ValueError("target_fpr must be in [0, 1)")

    sorted_non_owner = sorted(non_owner_scores)
    raw_index = int((1.0 - target_fpr) * len(sorted_non_owner)) - 1
    index = min(max(raw_index, 0), len(sorted_non_owner) - 1)
    threshold = sorted_non_owner[index]
    return {
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "owner_mean": float(sum(owner_scores) / len(owner_scores)),
        "non_owner_mean": float(sum(non_owner_scores) / len(non_owner_scores)),
        "owner_scores": [float(score) for score in owner_scores],
        "non_owner_scores": [float(score) for score in non_owner_scores],
    }


def verify_owner(
    *,
    owner_id: str,
    expected_codebook: list[int],
    recovered_codebook: list[int],
    negative_asr: float,
    competitor_scores: dict[str, float],
    threshold: float,
    margin: float,
    verification_path: Path,
    calibration_path: Path,
    negative_weight: float = 0.2,
) -> dict[str, object]:
    owner_score = compute_owner_score(
        expected_codebook=expected_codebook,
        recovered_codebook=recovered_codebook,
        negative_asr=negative_asr,
        negative_weight=negative_weight,
    )
    strongest_competitor = max(competitor_scores.values()) if competitor_scores else 0.0
    margin_value = owner_score - strongest_competitor
    accepted = owner_score >= threshold and margin_value >= margin
    ambiguity_flag = any(score >= threshold for score in competitor_scores.values())
    summary: dict[str, object] = {
        "owner_id": owner_id,
        "code_length": len(expected_codebook),
        "owner_score": owner_score,
        "negative_asr": negative_asr,
        "competitor_scores": competitor_scores,
        "margin": margin,
        "margin_value": margin_value,
        "decision": accepted,
        "ambiguity_flag": ambiguity_flag,
        "threshold": threshold,
        "recovered_codebook": recovered_codebook,
    }

    calibration = calibrate_threshold(
        owner_scores=[owner_score],
        non_owner_scores=list(competitor_scores.values()) if competitor_scores else [0.0],
        target_fpr=0.05,
    )
    write_json(verification_path, summary)
    write_json(calibration_path, calibration)
    return summary


def run_verification_from_checkpoint(
    *,
    checkpoint_path: Path,
    artifacts_path: Path,
    verification_path: Path,
    calibration_path: Path,
    decision_threshold: float,
    margin: float,
    competitor_owner_ids: list[str],
    seed: int,
) -> dict[str, object]:
    artifacts = load_owner_artifacts(artifacts_path)
    expected = [int(bit) for bit in artifacts["codebook"]]
    pos_queries = [
        torch.tensor(sample, dtype=torch.float32)
        for sample in artifacts["positive_queries"]
    ]
    neg_queries_raw = artifacts.get("negative_queries", [])
    neg_queries = [torch.tensor(sample, dtype=torch.float32) for sample in neg_queries_raw]

    model = _load_model_from_checkpoint(checkpoint_path)
    recovered = recover_codeword(query_model(model=model, queries=pos_queries))

    if neg_queries:
        neg_recovered = recover_codeword(query_model(model=model, queries=neg_queries))
        negative_matches = sum(int(bit == 1) for bit in neg_recovered)
        negative_asr = negative_matches / len(neg_recovered)
    else:
        negative_asr = 0.0

    competitor_scores: dict[str, float] = {}
    for competitor_id in competitor_owner_ids:
        competitor_codebook = generate_codebook(
            owner_id=competitor_id,
            code_length=len(expected),
            seed=seed,
        )
        competitor_scores[competitor_id] = compute_owner_score(
            expected_codebook=competitor_codebook,
            recovered_codebook=recovered,
            negative_asr=negative_asr,
            negative_weight=0.2,
        )

    return verify_owner(
        owner_id=str(artifacts["owner_id"]),
        expected_codebook=expected,
        recovered_codebook=recovered,
        negative_asr=negative_asr,
        competitor_scores=competitor_scores,
        threshold=decision_threshold,
        margin=margin,
        verification_path=verification_path,
        calibration_path=calibration_path,
    )


def verify_owner_from_checkpoint(
    *,
    checkpoint_path: Path,
    artifacts_path: Path,
    verification_path: Path,
    decision_threshold: float,
) -> dict[str, object]:
    summary = run_verification_from_checkpoint(
        checkpoint_path=checkpoint_path,
        artifacts_path=artifacts_path,
        verification_path=verification_path,
        calibration_path=verification_path.with_name("calibration_artifacts.json"),
        decision_threshold=decision_threshold,
        margin=0.0,
        competitor_owner_ids=[],
        seed=0,
    )
    return {
        "owner_id": summary["owner_id"],
        "code_length": summary["code_length"],
        "score": summary["owner_score"],
        "decision": summary["decision"],
        "threshold": summary["threshold"],
        "recovered_codebook": summary["recovered_codebook"],
    }
