import json
from pathlib import Path

from bbv.evaluation import summarize_outputs


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_privacy_summary_contains_leakage_metric(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run0"
    _write_json(
        run_dir / "verification_margin_summary.json",
        {
            "owner_id": "owner0",
            "owner_score": 0.62,
            "decision": True,
            "threshold": 0.5,
            "margin_value": 0.1,
            "competitor_scores": {"owner1": 0.4},
            "ambiguity_flag": False,
            "claim_type": "owner",
        },
    )
    _write_json(run_dir / "run_metadata.json", {"seed": 0})
    _write_json(
        run_dir / "allocation_assignments.json",
        {
            "round_assignments": [
                {
                    "assignments": {
                        "0": {"stats": {"class_coverage": 2, "skew_ratio": 0.9, "main_wm_alignment": 0.8, "privacy_penalty": 0.1}},
                        "1": {"stats": {"class_coverage": 8, "skew_ratio": 0.2, "main_wm_alignment": 0.1, "privacy_penalty": 0.1}},
                    }
                }
            ]
        },
    )

    summary = summarize_outputs(results_root=tmp_path / "runs", attacks_root=None)

    assert hasattr(summary, "privacy_leakage_auc")
    assert 0.0 <= summary.privacy_leakage_auc <= 1.0
