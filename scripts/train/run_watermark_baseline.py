"""Thin entrypoint for Phase 3 watermark baseline."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.federated import train_federated
from bbv.verification import verify_owner_from_checkpoint
from bbv.watermarking import (
    build_negative_queries,
    build_positive_queries,
    generate_codebook,
    save_owner_artifacts,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/train",
    config_name="watermark_baseline",
)
def main(cfg: DictConfig) -> None:
    train_result = train_federated(
        output_root=Path(cfg.output_root),
        seed=int(cfg.seed),
        dataset_name=str(cfg.dataset.name),
        model_name=str(cfg.model.name),
        num_classes=int(cfg.dataset.num_classes),
        num_clients=int(cfg.federated.num_clients),
        rounds=int(cfg.federated.rounds),
        participation_rate=float(cfg.federated.participation_rate),
        local_epochs=int(cfg.federated.local_epochs),
        batch_size=int(cfg.federated.batch_size),
        learning_rate=float(cfg.federated.learning_rate),
        samples_per_client=int(cfg.dataset.samples_per_client),
        allocation_enabled=bool(cfg.allocation.enabled),
        allocation_budget_ratio=float(cfg.allocation.budget_ratio),
        allocation_base_loss_weight=float(cfg.allocation.base_loss_weight),
    )

    codebook = generate_codebook(
        owner_id=str(cfg.owner.id),
        code_length=int(cfg.watermarking.code_length),
        seed=int(cfg.seed),
    )
    queries = build_positive_queries(codebook=codebook, seed=int(cfg.seed))
    negative_queries = build_negative_queries(codebook=codebook, seed=int(cfg.seed))

    artifacts_path = train_result.run_dir / "owner_artifacts.json"
    save_owner_artifacts(
        path=artifacts_path,
        owner_id=str(cfg.owner.id),
        codebook=codebook,
        queries=queries,
        negative_queries=negative_queries,
        wm_train_config={"task_weight": 1.0, "wm_weight": 0.2},
    )

    verification_path = train_result.run_dir / "verification_summary.json"
    summary = verify_owner_from_checkpoint(
        checkpoint_path=train_result.checkpoint_path,
        artifacts_path=artifacts_path,
        verification_path=verification_path,
        decision_threshold=float(cfg.verification.decision_threshold),
    )
    print(f"Watermark run directory: {train_result.run_dir}")
    print(f"Verification score: {summary['score']}")


if __name__ == "__main__":
    main()
