"""Phase 6 attack suite entrypoint."""

from __future__ import annotations

from pathlib import Path
import json

import hydra
from omegaconf import DictConfig

from bbv.attacks import run_attack
from bbv.federated.progress import progress_iterable
from bbv.verification import run_verification_from_checkpoint


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
    progress_steps = progress_iterable(
        ["resolve_checkpoint", "attack_model", "verify_attacked_model"],
        description="Attack pipeline",
        enabled=bool(cfg.progress.enabled),
        leave=True,
    )
    next(progress_steps)
    checkpoint_path = _resolve_checkpoint_path(Path(cfg.checkpoint))
    source_run_dir = checkpoint_path.parent
    artifacts_path = source_run_dir / "owner_artifacts.json"
    attack_config = {
        key: value
        for key, value in dict(cfg.attack).items()
        if key != "name"
    }
    next(progress_steps)
    result = run_attack(
        attack_name=str(cfg.attack.name),
        checkpoint_path=checkpoint_path,
        output_root=Path(cfg.output_root),
        seed=int(cfg.seed),
        dataset_name=str(cfg.dataset.name),
        attack_config=attack_config,
    )
    next(progress_steps)
    if artifacts_path.exists():
        owner_payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
        owner_id = str(owner_payload.get("owner_id", "owner0"))
        competitor_owner_ids = ["owner1" if owner_id != "owner1" else "owner0"]
        run_verification_from_checkpoint(
            checkpoint_path=result.attacked_checkpoint,
            artifacts_path=artifacts_path,
            verification_path=result.output_dir / "verification_after_attack.json",
            calibration_path=result.output_dir / "calibration_after_attack.json",
            decision_threshold=0.5,
            margin=0.05,
            competitor_owner_ids=competitor_owner_ids,
            seed=int(cfg.seed),
        )
    print(f"Attack output directory: {result.output_dir}")
    print(f"Attacked checkpoint: {result.attacked_checkpoint}")


if __name__ == "__main__":
    main()
