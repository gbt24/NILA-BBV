"""Export tables, figure, and summary from evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bbv.evaluation import EvaluationSummary


@dataclass(frozen=True)
class ReportBundle:
    main_table: Path
    ablation_table: Path
    robustness_table: Path
    main_figure: Path
    summary_report: Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_figure(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["accept", "ambiguity", "fpr", "fnr", "robust"]
    keys = [
        "acceptance_rate",
        "ambiguity_rate",
        "fpr",
        "fnr",
        "robustness_acceptance_rate",
    ]
    values = [max(0.0, min(1.0, float(metrics.get(key, 0.0)))) for key in keys]

    width = 640
    height = 300
    chart_left = 50
    chart_bottom = 250
    bar_width = 80
    gap = 30

    bars: list[str] = []
    for index, value in enumerate(values):
        x = chart_left + index * (bar_width + gap)
        bar_height = int(180 * value)
        y = chart_bottom - bar_height
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#2a7f62" />')
        bars.append(f'<text x="{x + bar_width / 2}" y="{chart_bottom + 18}" text-anchor="middle" font-size="12">{labels[index]}</text>')
        bars.append(f'<text x="{x + bar_width / 2}" y="{y - 6}" text-anchor="middle" font-size="11">{value:.2f}</text>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#f8faf8" />'
        '<text x="20" y="28" font-size="18" font-family="serif">Main Metrics Overview</text>'
        '<line x1="50" y1="250" x2="590" y2="250" stroke="#333" stroke-width="1" />'
        + "".join(bars)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _write_summary(path: Path, dataset: str, study: str, summary: EvaluationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Report Summary ({dataset} / {study})",
        "",
        "## Key Metrics",
        f"- acceptance_rate: {summary.metrics.get('acceptance_rate', 0.0):.4f}",
        f"- ambiguity_rate: {summary.metrics.get('ambiguity_rate', 0.0):.4f}",
        f"- fpr: {summary.metrics.get('fpr', 0.0):.4f}",
        f"- fnr: {summary.metrics.get('fnr', 0.0):.4f}",
        f"- robustness_acceptance_rate: {summary.metrics.get('robustness_acceptance_rate', 0.0):.4f}",
        "",
        f"main_rows: {len(summary.main_rows)}",
        f"ablation_rows: {len(summary.ablation_rows)}",
        f"robustness_rows: {len(summary.robustness_rows)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_report_bundle(
    *, dataset: str, study: str, summary: EvaluationSummary, output_root: Path
) -> ReportBundle:
    output_root = Path(output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    summaries_dir = output_root / "summaries"

    prefix = f"{dataset}-{study}"
    main_table = tables_dir / f"{prefix}-main-results.csv"
    ablation_table = tables_dir / f"{prefix}-ablation-results.csv"
    robustness_table = tables_dir / f"{prefix}-robustness-results.csv"
    main_figure = figures_dir / f"{prefix}-main-figure.svg"
    summary_report = summaries_dir / f"{prefix}-summary.md"

    _write_csv(main_table, summary.main_rows)
    _write_csv(ablation_table, summary.ablation_rows)
    _write_csv(robustness_table, summary.robustness_rows)
    _write_main_figure(main_figure, summary.metrics)
    _write_summary(summary_report, dataset, study, summary)

    return ReportBundle(
        main_table=main_table,
        ablation_table=ablation_table,
        robustness_table=robustness_table,
        main_figure=main_figure,
        summary_report=summary_report,
    )
