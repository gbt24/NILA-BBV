"""Post-dispute commitment audit: verify pre-training hash and run black-box verification.

Usage:
    uv run python scripts/eval/run_commitment_audit.py \
        checkpoint=path/to/checkpoint.pt \
        commitment=path/to/pre_training_commitment.json \
        artifacts=path/to/owner_artifacts.json \
        revealed_seed=0 \
        decision_threshold=0.5 \
        decision_margin=0.05
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.verification.baseline import run_verification_from_checkpoint
from bbv.watermarking.codebook import compute_codebook_hash, generate_codebook
from bbv.utils.io import write_json


def verify_commitment(commitment_path: Path, revealed_seed: int, revealed_code_length: int) -> dict:
    with open(commitment_path, encoding="utf-8") as f:
        commitment = json.load(f)
    committed_hash = str(commitment["commitment_hash"])
    committed_owner_id = str(commitment["owner_id"])
    committed_timestamp = str(commitment.get("timestamp_utc", "unknown"))

    codebook = generate_codebook(committed_owner_id, revealed_code_length, revealed_seed)
    revealed_hash = compute_codebook_hash(codebook)

    match = revealed_hash == committed_hash
    status = "VERIFIED" if match else "VIOLATION"
    return {
        "commitment_timestamp": committed_timestamp,
        "owner_id": committed_owner_id,
        "committed_hash": committed_hash,
        "revealed_hash": revealed_hash,
        "hash_match": match,
        "status": status,
        "artifact_key": f"{committed_owner_id}-{revealed_seed}-{revealed_code_length}",
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/train",
    config_name="watermark_baseline",
)
def main(cfg: DictConfig) -> None:
    checkpoint = Path(str(cfg.get("checkpoint", "")))
    commitment = Path(str(cfg.get("commitment", "")))
    artifacts = Path(str(cfg.get("artifacts", "")))
    revealed_seed = int(cfg.get("revealed_seed", 0))
    revealed_code_length = int(cfg.get("revealed_code_length", 64))
    tau = float(cfg.get("decision_threshold", 0.5))
    gamma = float(cfg.get("decision_margin", 0.05))
    competitor_ids = list(cfg.get("competitor_owner_ids", ["owner1", "owner2", "owner3", "owner4"]))

    commitment_result = verify_commitment(
        commitment, revealed_seed, revealed_code_length
    )
    print(f"Commitment audit: {commitment_result['status']}")
    print(f"  Timestamp:  {commitment_result['commitment_timestamp']}")
    print(f"  Committed:  {commitment_result['committed_hash'][:16]}...")
    print(f"  Revealed:   {commitment_result['revealed_hash'][:16]}...")

    if not commitment_result["hash_match"]:
        print("ABORT: commitment verification failed. Possible post-hoc fabrication.")
        return

    verification = run_verification_from_checkpoint(
        checkpoint_path=checkpoint,
        artifacts_path=artifacts,
        decision_threshold=tau,
        margin=gamma,
        competitor_owner_ids=competitor_ids,
        seed=revealed_seed,
        hard_label_only=True,
        device=str(cfg.device),
    )

    output_dir = Path(cfg.get("output_root", checkpoint.parent))
    audit_path = output_dir / "commitment_audit_report.json"
    report = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "commitment": {
            "path": str(commitment),
            "result": commitment_result,
        },
        "verification": {
            "owner_score": float(verification["owner_score"]),
            "margin": float(verification["margin_value"]),
            "decision": bool(verification["decision"]),
            "threshold": tau,
            "margin_threshold": gamma,
        },
    }
    write_json(audit_path, report)

    print(f"\nVerification result:")
    print(f"  Owner score:   {verification['owner_score']:.4f}")
    print(f"  Margin:        {verification['margin_value']:.4f}")
    print(f"  Decision:      {'ACCEPT' if verification['decision'] else 'REJECT'}")
    print(f"\nFull audit report: {audit_path}")


if __name__ == "__main__":
    main()
