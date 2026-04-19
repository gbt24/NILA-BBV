"""Thin entrypoint for the Phase 0 smoke training run."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.federated.runner import run_smoke_from_config


@hydra.main(version_base=None, config_path="../../configs/train", config_name="smoke")
def main(cfg: DictConfig) -> None:
    result = run_smoke_from_config(cfg)
    print(f"Smoke run summary: {Path(result.summary_path)}")


if __name__ == "__main__":
    main()
