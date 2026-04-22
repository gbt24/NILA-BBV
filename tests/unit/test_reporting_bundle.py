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


def test_export_report_bundle_writes_tradeoff_figure(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
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
        dataset="cifar10", study="main", summary=summary, output_root=output_root
    )

    assert hasattr(bundle, "tradeoff_figure")
    assert (output_root / "figures" / "cifar10-main-tradeoff-figure.svg").exists()


def test_export_report_bundle_summary_includes_all_key_metrics(tmp_path: Path) -> None:
    summary = EvaluationSummary(
        main_rows=[{"run_id": "r0", "owner_score": 0.7, "decision": True}],
        ablation_rows=[{"run_id": "r0", "allocation_enabled": True}],
        robustness_rows=[{"attack": "finetune", "decision": False}],
        metrics={
            "acceptance_rate": 1.0,
            "ambiguity_rate": 0.1,
            "fpr": 0.2,
            "fnr": 0.3,
            "false_claim_acceptance_rate": 0.4,
            "robustness_acceptance_rate": 0.5,
        },
    )

    bundle = export_report_bundle(
        dataset="cifar10", study="main", summary=summary, output_root=tmp_path
    )

    summary_text = bundle.summary_report.read_text(encoding="utf-8")

    assert "acceptance_rate" in summary_text
    assert "ambiguity_rate" in summary_text
    assert "fpr" in summary_text
    assert "fnr" in summary_text
    assert "false_claim_acceptance_rate" in summary_text
    assert "robustness_acceptance_rate" in summary_text
