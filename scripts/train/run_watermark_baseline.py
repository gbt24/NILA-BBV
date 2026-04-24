"""Thin entrypoint for Phase 3 watermark baseline."""

from __future__ import annotations

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


@hydra.main(
    version_base=None,
    config_path="../../configs/train",
    config_name="watermark_baseline",
)
def main(cfg: DictConfig) -> None:
    watermark_hook = WatermarkHook(
        owner_id=str(cfg.owner.id),
        code_length=int(cfg.watermarking.code_length),
        wm_weight=float(cfg.watermarking.wm_weight),
        seed=int(cfg.seed),
    )
    train_result = train_federated(
        output_root=Path(cfg.output_root),
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
    artifacts_path = train_result.run_dir / "owner_artifacts.json"
    artifacts = load_owner_artifacts(artifacts_path)
    wm_train_config = dict(artifacts.get("wm_train_config", {}))
    commitment_path = train_result.run_dir / "owner_commitment.json"
    save_commitment_record(
        commitment_path,
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


if __name__ == "__main__":
    main()
