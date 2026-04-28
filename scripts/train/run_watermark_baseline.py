"""Thin entrypoint for Phase 3 watermark baseline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.federated import train_federated
from bbv.federated.hooks import WatermarkHook
from bbv.verification import verify_owner_from_checkpoint
from bbv.watermarking import (
    build_commitment_record,
    load_owner_artifacts,
    save_commitment_record,
)
from bbv.utils.io import write_json


@hydra.main(
    version_base=None,
    config_path="../../configs/train",
    config_name="watermark_baseline",
)
def main(cfg: DictConfig) -> None:
    wm_enabled = bool(cfg.watermarking.get("enabled", True))
    codebook_type = str(cfg.watermarking.get("codebook_type", "multi-bit"))
    
    if wm_enabled:
        watermark_hook = WatermarkHook(
            owner_id=str(cfg.owner.id),
            code_length=int(cfg.watermarking.code_length),
            wm_weight=float(cfg.watermarking.wm_weight),
            seed=int(cfg.seed),
            codebook_type=codebook_type,
        )
    else:
        watermark_hook = None

    output_dir = Path(cfg.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if wm_enabled:
        pre_commitment_path = output_dir / f"pre_training_commitment_seed{int(cfg.seed)}.json"
        write_json(pre_commitment_path, {
            "commitment_hash": hashlib.sha256(
                "".join(str(b) for b in watermark_hook.codebook).encode("utf-8")
            ).hexdigest(),
            "owner_id": str(cfg.owner.id),
            "seed": int(cfg.seed),
            "code_length": int(cfg.watermarking.code_length),
            "codebook_type": codebook_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    train_result = train_federated(
        output_root=output_dir,
        seed=int(cfg.seed),
        dataset_name=str(cfg.dataset.name),
        model_name=str(cfg.dataset.get("model_name", cfg.model.name)),
        num_classes=int(cfg.dataset.num_classes),
        num_clients=int(cfg.federated.num_clients),
        rounds=int(cfg.federated.rounds),
        participation_rate=float(cfg.federated.participation_rate),
        local_epochs=int(cfg.federated.local_epochs),
        batch_size=int(cfg.federated.batch_size),
        learning_rate=float(cfg.federated.learning_rate),
        samples_per_client=int(cfg.dataset.samples_per_client),
        partition_type=str(cfg.dataset.partition_type),
        concentration=float(cfg.dataset.concentration),
        shards_per_client=int(cfg.dataset.shards_per_client),
        quantity_sigma=float(cfg.dataset.quantity_sigma),
        watermark_hook=watermark_hook,
        allocation_enabled=bool(cfg.allocation.enabled),
        allocation_budget_ratio=float(cfg.allocation.budget_ratio),
        allocation_base_loss_weight=float(cfg.allocation.base_loss_weight),
        progress_enabled=bool(cfg.progress.enabled),
        device=str(cfg.device),
    )
    if wm_enabled:
        artifacts_path = train_result.run_dir / "owner_artifacts.json"
        artifacts = load_owner_artifacts(artifacts_path)
        wm_train_config = dict(artifacts.get("wm_train_config", {}))
        save_commitment_record(
            train_result.run_dir / "owner_commitment.json",
            build_commitment_record(
                owner_id=str(cfg.owner.id),
                seed=int(cfg.seed),
                codebook=[int(bit) for bit in artifacts["codebook"]],
                config=wm_train_config,
            ),
        )
        verification_path = train_result.run_dir / "verification_summary.json"
        summary = verify_owner_from_checkpoint(
            checkpoint_path=train_result.best_checkpoint_path,
            artifacts_path=artifacts_path,
            verification_path=verification_path,
            decision_threshold=float(cfg.verification.decision_threshold),
        )
        print(f"Watermark run directory: {train_result.run_dir}")
        print(f"Verification score: {summary['score']}")
        print(f"Codebook hash: {artifacts['codebook_hash']}")
    else:
        print(f"No-watermark run directory: {train_result.run_dir}")


if __name__ == "__main__":
    main()
