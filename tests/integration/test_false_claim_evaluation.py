import json
from pathlib import Path

from bbv.evaluation import summarize_outputs


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_evaluation_reports_false_claim_acceptance_rate(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run0"
    _write_json(
        run_dir / "verification_margin_summary.json",
        {
            "owner_id": "owner0",
            "owner_score": 0.62,
            "decision": True,
            "threshold": 0.5,
            "margin_value": 0.1,
            "competitor_scores": {"owner1": 0.55},
            "ambiguity_flag": True,
        },
    )
    _write_json(run_dir / "run_metadata.json", {"seed": 0})

    summary = summarize_outputs(results_root=tmp_path / "runs", attacks_root=None)

    assert "false_claim_acceptance_rate" in summary.metrics
    assert summary.metrics["false_claim_acceptance_rate"] >= 0.0
