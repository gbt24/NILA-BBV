"""Phase 7 report builder entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from bbv.evaluation import summarize_outputs
from bbv.reporting import export_report_bundle


@hydra.main(version_base=None, config_path="../../configs/report", config_name="main")
def main(cfg: DictConfig) -> None:
    outputs_dir = Path(cfg.outputs_dir)
    attacks_dir = Path(cfg.attacks_dir)
    report_root = Path(cfg.get("bundle_dir", cfg.report_root))

    summary = summarize_outputs(results_root=outputs_dir, attacks_root=attacks_dir)
    bundle = export_report_bundle(
        dataset=str(cfg.dataset.name),
        study=str(cfg.study.name),
        summary=summary,
        output_root=report_root,
    )
    print(f"Main table: {bundle.main_table}")
    print(f"Ablation table: {bundle.ablation_table}")
    print(f"Robustness table: {bundle.robustness_table}")
    print(f"Main figure: {bundle.main_figure}")
    print(f"Tradeoff figure: {bundle.tradeoff_figure}")
    print(f"Summary report: {bundle.summary_report}")


if __name__ == "__main__":
    main()
