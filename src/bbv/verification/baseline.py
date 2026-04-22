"""Black-box verification with score, margin, and calibration."""

from __future__ import annotations

from pathlib import Path

import torch

from bbv.models import build_model
from bbv.utils.io import write_json
from bbv.verification.calibration import calibrate_threshold
from bbv.verification.query import (
    batched_query_model,
    batched_query_model_logits,
    query_model,
    query_model_logits,
)
from bbv.watermarking import generate_codebook, load_owner_artifacts


def _load_model_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    model_name = str(checkpoint.get("model_name", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", state_dict[next(reversed(state_dict))].shape[0]))
    input_shape = tuple(checkpoint.get("input_shape", (3, 32, 32)))
    model = build_model(model_name, num_classes=num_classes, input_shape=input_shape)
    model.load_state_dict(state_dict)
    setattr(model, "_bbv_input_shape", input_shape)
    model.eval()
    return model


def recover_codeword(predicted_labels: list[int]) -> list[int]:
    return [int(label % 2) for label in predicted_labels]


def recover_codeword_from_logits(logits_list: list[torch.Tensor]) -> list[int]:
    recovered: list[int] = []
    for logits in logits_list:
        even_logits = torch.logsumexp(logits[0::2], dim=0)
        odd_logits = torch.logsumexp(logits[1::2], dim=0)
        parity_logits = torch.stack([even_logits, odd_logits], dim=0)
        recovered.append(int(parity_logits.argmax().item()))
    return recovered


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


def compute_negative_asr(negative_codebook: list[int], neg_recovered: list[int]) -> float:
    if not neg_recovered:
        return 0.0
    expected = negative_codebook[: len(neg_recovered)]
    matches = sum(int(left == right) for left, right in zip(expected, neg_recovered, strict=False))
    return matches / len(neg_recovered)


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
    query_budget: int | None = None,
    hard_label_only: bool = True,
    queried_positive_count: int = 0,
    queried_negative_count: int = 0,
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
    ambiguity_flag = any(
        score >= threshold or (owner_score >= threshold and score >= owner_score - margin)
        for score in competitor_scores.values()
    )
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
        "query_budget": query_budget,
        "hard_label_only": hard_label_only,
        "queried_positive_count": queried_positive_count,
        "queried_negative_count": queried_negative_count,
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
    query_budget: int | None = None,
    hard_label_only: bool = True,
    batch_size: int = 16,
    negative_weight: float = 0.2,
    expected_owner_id: str | None = None,
) -> dict[str, object]:
    if query_budget is not None and query_budget <= 0:
        raise ValueError("query_budget must be greater than 0")

    artifacts = load_owner_artifacts(artifacts_path)
    if expected_owner_id is not None and str(artifacts["owner_id"]) != expected_owner_id:
        raise ValueError("owner_id does not match verification artifacts")
    expected = [int(bit) for bit in artifacts["codebook"]]
    pos_queries = [
        torch.tensor(sample, dtype=torch.float32)
        for sample in artifacts["positive_queries"]
    ]
    neg_queries_raw = artifacts.get("negative_queries", [])
    neg_queries = [torch.tensor(sample, dtype=torch.float32) for sample in neg_queries_raw]
    negative_codebook = [1 - bit for bit in expected]

    model = _load_model_from_checkpoint(checkpoint_path)

    if query_budget is None:
        positive_budget = None
        negative_budget = None
    elif neg_queries:
        reserved_positive_budget = max(1, query_budget // 2)
        positive_budget = min(len(pos_queries), reserved_positive_budget)
        negative_budget = min(len(neg_queries), max(query_budget - positive_budget, 0))
        positive_budget = min(
            len(pos_queries),
            positive_budget + max(query_budget - positive_budget - negative_budget, 0),
        )
    else:
        positive_budget = min(len(pos_queries), query_budget)
        negative_budget = 0

    if hard_label_only:
        recovered = recover_codeword(
            batched_query_model(
                model=model,
                queries=pos_queries,
                batch_size=batch_size,
                max_queries=positive_budget,
            )
        )
    else:
        recovered = recover_codeword_from_logits(
            batched_query_model_logits(
                model=model,
                queries=pos_queries,
                batch_size=batch_size,
                max_queries=positive_budget,
            )
        )

    expected_for_scoring = expected[: len(recovered)]
    queried_positive_count = len(recovered)

    if neg_queries:
        if hard_label_only:
            neg_recovered = recover_codeword(
                batched_query_model(
                    model=model,
                    queries=neg_queries,
                    batch_size=batch_size,
                    max_queries=negative_budget,
                )
            )
        else:
            neg_recovered = recover_codeword_from_logits(
                batched_query_model_logits(
                    model=model,
                    queries=neg_queries,
                    batch_size=batch_size,
                    max_queries=negative_budget,
                )
            )
        queried_negative_count = len(neg_recovered)
        negative_asr = compute_negative_asr(negative_codebook, neg_recovered)
    else:
        queried_negative_count = 0
        negative_asr = 0.0

    competitor_scores: dict[str, float] = {}
    for competitor_id in competitor_owner_ids:
        competitor_full_codebook = generate_codebook(
            owner_id=competitor_id,
            code_length=len(expected),
            seed=seed,
        )
        competitor_codebook = competitor_full_codebook[: len(expected_for_scoring)]
        competitor_negative_codebook = [1 - bit for bit in competitor_full_codebook]
        competitor_negative_asr = compute_negative_asr(
            competitor_negative_codebook,
            neg_recovered if neg_queries else [],
        )
        competitor_scores[competitor_id] = compute_owner_score(
            expected_codebook=competitor_codebook,
            recovered_codebook=recovered,
            negative_asr=competitor_negative_asr,
            negative_weight=negative_weight,
        )

    return verify_owner(
        owner_id=str(artifacts["owner_id"]),
        expected_codebook=expected_for_scoring,
        recovered_codebook=recovered,
        negative_asr=negative_asr,
        competitor_scores=competitor_scores,
        threshold=decision_threshold,
        margin=margin,
        verification_path=verification_path,
        calibration_path=calibration_path,
        negative_weight=negative_weight,
        query_budget=query_budget,
        hard_label_only=hard_label_only,
        queried_positive_count=queried_positive_count,
        queried_negative_count=queried_negative_count,
    )


def verify_owner_from_checkpoint(
    *,
    checkpoint_path: Path,
    artifacts_path: Path,
    verification_path: Path,
    decision_threshold: float,
    expected_owner_id: str | None = None,
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
        expected_owner_id=expected_owner_id,
    )
    return {
        "owner_id": summary["owner_id"],
        "code_length": summary["code_length"],
        "score": summary["owner_score"],
        "decision": summary["decision"],
        "threshold": summary["threshold"],
        "recovered_codebook": summary["recovered_codebook"],
    }
