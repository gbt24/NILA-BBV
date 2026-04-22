"""Phase 5 verification entrypoint for margin and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.verification import run_verification_from_checkpoint


def _find_latest_run_dir(output_root: Path, owner_id: str | None = None) -> Path:
    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run directories found under {output_root}")
    if owner_id is not None:
        filtered: list[Path] = []
        for path in candidates:
            artifacts_path = path / "owner_artifacts.json"
            if not artifacts_path.exists():
                continue
            payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
            if str(payload.get("owner_id")) == owner_id:
                filtered.append(path)
        candidates = filtered
        if not candidates:
            raise FileNotFoundError(
                f"no run directories found under {output_root} for owner_id={owner_id}"
            )
    def sort_key(path: Path) -> float:
        best_checkpoint = path / "best_checkpoint.pt"
        checkpoint = path / "checkpoint.pt"
        if best_checkpoint.exists():
            return best_checkpoint.stat().st_mtime
        if checkpoint.exists():
            return checkpoint.stat().st_mtime
        return path.stat().st_mtime

    candidates.sort(key=sort_key)
    return candidates[-1]


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="main")
def main(cfg: DictConfig) -> None:
    output_root = Path(cfg.output_root)
    run_dir = _find_latest_run_dir(output_root, owner_id=str(cfg.owner.id))
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint.pt"
    summary = run_verification_from_checkpoint(
        checkpoint_path=checkpoint_path,
        artifacts_path=run_dir / "owner_artifacts.json",
        verification_path=run_dir / "verification_margin_summary.json",
        calibration_path=run_dir / "calibration_artifacts.json",
        decision_threshold=float(cfg.verification.decision_threshold),
        margin=float(cfg.verification.margin),
        competitor_owner_ids=list(cfg.verification.competitor_owner_ids),
        seed=int(cfg.seed),
        query_budget=cfg.verification.get("query_budget"),
        hard_label_only=bool(cfg.verification.get("hard_label_only", True)),
        batch_size=int(cfg.verification.get("batch_size", 16)),
        expected_owner_id=str(cfg.owner.id),
    )
    print(f"Verification run directory: {run_dir}")
    print(f"Owner score: {summary['owner_score']}")


if __name__ == "__main__":
    main()
