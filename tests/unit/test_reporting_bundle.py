from pathlib import Path

from bbv.evaluation import EvaluationSummary
from bbv.reporting import export_report_bundle


def test_export_report_bundle_writes_figure_table_and_summary(tmp_path: Path) -> None:
    summary = EvaluationSummary(
        main_rows=[{"run_id": "r0", "owner_score": 0.7, "decision": True}],
        ablation_rows=[{"run_id": "r0", "allocation_enabled": True}],
        robustness_rows=[{"attack": "finetune", "decision": False}],
        metrics={
            "acceptance_rate": 1.0,
            "ambiguity_rate": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "false_claim_acceptance_rate": 0.0,
            "robustness_acceptance_rate": 0.0,
        },
    )
    bundle = export_report_bundle(
        dataset="cifar10", study="main", summary=summary, output_root=tmp_path
    )
    assert bundle.main_table.exists()
    assert bundle.main_figure.exists()
    assert bundle.summary_report.exists()
    assert (tmp_path / "figures" / "cifar10-main-tradeoff-figure.svg").exists()
