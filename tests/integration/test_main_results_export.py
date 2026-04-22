import json
from pathlib import Path

from bbv.evaluation import summarize_outputs
from bbv.reporting import export_report_bundle


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_evaluation_and_reporting_export_three_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_json(
        run_dir / "verification_margin_summary.json",
        {
            "owner_id": "owner0",
            "owner_score": 0.71,
            "decision": True,
            "threshold": 0.5,
            "margin_value": 0.21,
            "competitor_scores": {"owner1": 0.2},
            "ambiguity_flag": False,
        },
    )
    _write_json(run_dir / "run_metadata.json", {"seed": 0})

    attack_dir = tmp_path / "attacks" / "run-atk"
    _write_json(attack_dir / "attack_log.json", {"attack": "finetune"})
    _write_json(attack_dir / "verification_after_attack.json", {"owner_score": 0.45, "decision": False})

    summary = summarize_outputs(results_root=tmp_path / "runs", attacks_root=tmp_path / "attacks")
    bundle = export_report_bundle(dataset="cifar10", study="main", summary=summary, output_root=tmp_path / "outputs")

    assert bundle.main_table.exists()
    assert bundle.ablation_table.exists()
    assert bundle.robustness_table.exists()
    assert bundle.main_figure.exists()
    assert bundle.tradeoff_figure.exists()
    assert bundle.summary_report.exists()
    assert bundle.main_table.name == "cifar10-main-main-results.csv"
    assert bundle.ablation_table.name == "cifar10-main-ablation-results.csv"
    assert bundle.robustness_table.name == "cifar10-main-robustness-results.csv"
    assert bundle.main_figure.name == "cifar10-main-main-figure.svg"
    assert bundle.tradeoff_figure.name == "cifar10-main-tradeoff-figure.svg"
    assert bundle.summary_report.name == "cifar10-main-summary.md"

    summary_text = bundle.summary_report.read_text(encoding="utf-8")
    assert "acceptance_rate" in summary_text
    assert "ambiguity_rate" in summary_text
    assert "fpr" in summary_text
    assert "fnr" in summary_text
    assert "false_claim_acceptance_rate" in summary_text
    assert "robustness_acceptance_rate" in summary_text
