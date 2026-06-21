"""Build anti-false-claim protocol summaries from verification artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from bbv.utils.io import write_json
from bbv.verification.protocol import build_protocol_summary


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"summary file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "owner_score" not in payload:
        raise ValueError(f"summary file lacks owner_score: {path}")
    return payload


def _paths(raw: object) -> list[Path]:
    if raw is None:
        return []
    values = OmegaConf.to_container(raw, resolve=True) if not isinstance(raw, list) else raw
    return [Path(str(value)) for value in values]


def _scores(paths: list[Path]) -> list[float]:
    return [float(_load_summary(path)["owner_score"]) for path in paths]


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="false_claim_protocol")
def main(cfg: DictConfig) -> None:
    if cfg.owner_summary_path is None:
        raise ValueError("owner_summary_path is required")

    owner_summary_path = Path(str(cfg.owner_summary_path))
    owner_summary = _load_summary(owner_summary_path)
    owner_score = float(owner_summary["owner_score"])
    competitor_scores = {
        str(owner_id): float(score)
        for owner_id, score in dict(owner_summary.get("competitor_scores", {})).items()
    }
    calibration_owner_scores = _scores(_paths(cfg.calibration_owner_summary_paths))
    if not calibration_owner_scores:
        calibration_owner_scores = [owner_score]
    calibration_non_owner_scores = _scores(_paths(cfg.calibration_non_owner_summary_paths))
    false_claim_scores = _scores(_paths(cfg.false_claim_summary_paths))

    summary = build_protocol_summary(
        owner_id=str(owner_summary.get("owner_id", "owner0")),
        owner_score=owner_score,
        competitor_scores=competitor_scores,
        calibration_owner_scores=calibration_owner_scores,
        calibration_non_owner_scores=calibration_non_owner_scores,
        false_claim_scores=false_claim_scores,
        target_fpr=float(cfg.target_fpr),
        margin=float(cfg.margin),
        query_budget=owner_summary.get("query_budget"),
        hard_label_only=bool(owner_summary.get("hard_label_only", True)),
        smoke_only=bool(cfg.smoke_only),
    )
    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary)
    print(f"False-claim protocol summary: {output_path}")


if __name__ == "__main__":
    main()
