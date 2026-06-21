"""Verification pipeline modules."""

from bbv.verification.calibration import calibrate_threshold
from bbv.verification.baseline import (
    compute_negative_asr,
    compute_owner_score,
    recover_codeword,
    run_verification_from_checkpoint,
    verify_owner,
    verify_owner_from_checkpoint,
)
from bbv.verification.protocol import (
    build_protocol_summary,
    compute_false_claim_success_rate,
    decide_claim,
)
from bbv.verification.query import batched_query_model, query_model

__all__ = [
    "calibrate_threshold",
    "batched_query_model",
    "build_protocol_summary",
    "compute_negative_asr",
    "compute_false_claim_success_rate",
    "compute_owner_score",
    "decide_claim",
    "query_model",
    "recover_codeword",
    "run_verification_from_checkpoint",
    "verify_owner",
    "verify_owner_from_checkpoint",
]
