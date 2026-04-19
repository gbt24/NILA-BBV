"""Phase 5 verification entrypoint for margin and calibration."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.verification import run_verification_from_checkpoint


def _find_latest_run_dir(output_root: Path) -> Path:
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run directories found under {output_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="verification")
def main(cfg: DictConfig) -> None:
    output_root = Path(cfg.output_root)
    run_dir = _find_latest_run_dir(output_root)
    summary = run_verification_from_checkpoint(
        checkpoint_path=run_dir / "checkpoint.pt",
        artifacts_path=run_dir / "owner_artifacts.json",
        verification_path=run_dir / "verification_margin_summary.json",
        calibration_path=run_dir / "calibration_artifacts.json",
        decision_threshold=float(cfg.verification.decision_threshold),
        margin=float(cfg.verification.margin),
        competitor_owner_ids=list(cfg.verification.competitor_owner_ids),
        seed=int(cfg.seed),
    )
    print(f"Verification run directory: {run_dir}")
    print(f"Owner score: {summary['owner_score']}")


if __name__ == "__main__":
    main()
