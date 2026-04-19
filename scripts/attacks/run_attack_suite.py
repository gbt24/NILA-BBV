"""Phase 6 attack suite entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.attacks import run_attack


@hydra.main(version_base=None, config_path="../../configs/attacks", config_name="suite")
def main(cfg: DictConfig) -> None:
    result = run_attack(
        attack_name=str(cfg.attack),
        checkpoint_path=Path(cfg.checkpoint),
        output_root=Path(cfg.output_root),
        seed=int(cfg.seed),
        dataset_name=str(cfg.dataset),
    )
    print(f"Attack output directory: {result.output_dir}")
    print(f"Attacked checkpoint: {result.attacked_checkpoint}")


if __name__ == "__main__":
    main()
