import json
from pathlib import Path

from bbv.evaluation import summarize_outputs
from bbv.reporting import export_report_bundle


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_summarize_outputs_and_export_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fedavg-run-1"
    _write_json(
        run_dir / "verification_margin_summary.json",
        {
            "owner_id": "owner0",
            "owner_score": 0.72,
            "decision": True,
            "threshold": 0.5,
            "margin_value": 0.2,
            "competitor_scores": {"owner1": 0.22, "owner2": 0.31},
        },
    )
    _write_json(
        run_dir / "run_metadata.json",
        {"allocation": {"enabled": True}, "seed": 0},
    )

    attack_dir = tmp_path / "attacks" / "finetune-run-1"
    _write_json(attack_dir / "attack_log.json", {"attack": "finetune"})
    _write_json(
        attack_dir / "verification_after_attack.json",
        {"owner_score": 0.44, "decision": False},
    )

    summary = summarize_outputs(results_root=tmp_path / "runs", attacks_root=tmp_path / "attacks")
    assert len(summary.main_rows) == 1
    assert len(summary.robustness_rows) == 1
    assert "ambiguity_rate" in summary.metrics

    bundle = export_report_bundle(
        dataset="cifar10",
        study="main",
        summary=summary,
        output_root=tmp_path / "outputs",
    )
    assert bundle.main_table.exists()
    assert bundle.main_figure.exists()
    assert bundle.summary_report.exists()
