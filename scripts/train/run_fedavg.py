"""Thin entrypoint for the Phase 2 FedAvg baseline run."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.federated import train_federated


@hydra.main(version_base=None, config_path="../../configs/train", config_name="fedavg")
def main(cfg: DictConfig) -> None:
    result = train_federated(
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
        partition_type=str(cfg.dataset.partition_type),
        concentration=float(cfg.dataset.concentration),
        shards_per_client=int(cfg.dataset.shards_per_client),
        quantity_sigma=float(cfg.dataset.quantity_sigma),
        progress_enabled=bool(cfg.progress.enabled),
    )
    print(f"FedAvg run directory: {Path(result.run_dir)}")


if __name__ == "__main__":
    main()
