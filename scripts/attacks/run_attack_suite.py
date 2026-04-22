"""Phase 6 attack suite entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.attacks import run_attack


def _resolve_checkpoint_path(checkpoint: Path) -> Path:
    if checkpoint.exists():
        return checkpoint
    if checkpoint.as_posix().endswith("outputs/runs/latest/checkpoint.pt"):
        runs_root = checkpoint.parent.parent
        candidates = [
            path
            for path in runs_root.iterdir()
            if path.is_dir() and ((path / "best_checkpoint.pt").exists() or (path / "checkpoint.pt").exists())
        ] if runs_root.exists() else []
        if not candidates:
            raise FileNotFoundError(f"no run directories with checkpoints found under {runs_root}")
        candidates.sort(
            key=lambda path: (path / "best_checkpoint.pt").stat().st_mtime
            if (path / "best_checkpoint.pt").exists()
            else (path / "checkpoint.pt").stat().st_mtime
        )
        latest = candidates[-1]
        best_checkpoint = latest / "best_checkpoint.pt"
        return best_checkpoint if best_checkpoint.exists() else latest / "checkpoint.pt"
    return checkpoint


@hydra.main(version_base=None, config_path="../../configs/attacks", config_name="main")
def main(cfg: DictConfig) -> None:
    checkpoint_path = _resolve_checkpoint_path(Path(cfg.checkpoint))
    attack_config = {
        key: value
        for key, value in dict(cfg.attack).items()
        if key != "name"
    }
    result = run_attack(
        attack_name=str(cfg.attack.name),
        checkpoint_path=checkpoint_path,
        output_root=Path(cfg.output_root),
        seed=int(cfg.seed),
        dataset_name=str(cfg.dataset.name),
        attack_config=attack_config,
    )
    print(f"Attack output directory: {result.output_dir}")
    print(f"Attacked checkpoint: {result.attacked_checkpoint}")


if __name__ == "__main__":
    main()
