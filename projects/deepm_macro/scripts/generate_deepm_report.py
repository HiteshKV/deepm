#!/usr/bin/env python
"""Generate a PDF technical report for a finalized DeePM backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from deepm._paths import PROJECT_ROOT


REQUIRED_DIAGNOSTIC_IMAGES = [
    "pnl_cumulative.png",
    "annual_sharpe.png",
    "group_gross_pnl.png",
    "group_net_pnl.png",
]


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.92, title, fontsize=20, weight="bold")
    y = 0.84
    for line in lines:
        fig.text(0.06, y, line, fontsize=11, va="top")
        y -= 0.045
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, max_rows: int = 24) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=16)
    shown = frame.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    table = ax.table(
        cellText=shown.astype(str).values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    image = mpimg.imread(image_path)
    ax.imshow(image)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _selection_plot_page(pdf: PdfPages, selection: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, (test_start, group) in zip(axes, selection.groupby("test_start")):
        ax.bar(group["rank"].astype(str), group["valid_loss_best"])
        ax.set_title(f"{test_start} top 10 validation")
        ax.set_xlabel("Rank")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation Sharpe")
    fig.suptitle("Selected Seed Validation Ranking", fontsize=16, weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _drawdown_page(pdf: PdfPages, pnl_path: Path) -> None:
    pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
    wealth = pnl["net_wealth_index"]
    drawdown = wealth / wealth.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    drawdown.plot(ax=ax, color="#8f1d1d")
    ax.set_title("Net Drawdown", loc="left", fontsize=16, weight="bold")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def validate_inputs(diagnostics_dir: Path, metrics_csv: Path, report_dir: Path) -> None:
    required = [diagnostics_dir / "metrics.json", diagnostics_dir / "pnl_curves.csv", metrics_csv]
    required.extend(diagnostics_dir / name for name in REQUIRED_DIAGNOSTIC_IMAGES)
    required.extend([report_dir / "selection_summary.csv", report_dir / "window_model_selection.csv"])
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required report inputs:\n" + "\n".join(str(path) for path in missing))


def generate_report(
    model_backtest: str,
    metrics_csv: Path,
    run: str,
    arch: str,
    top_n: int,
    output: Path,
) -> Path:
    diagnostics_dir = PROJECT_ROOT / "backtest_diagnostics" / model_backtest
    report_dir = PROJECT_ROOT / "reports" / f"{run.replace('-', '_')}_k{top_n}_20260625"
    validate_inputs(diagnostics_dir, metrics_csv, report_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    selection_summary = pd.read_csv(report_dir / "selection_summary.csv")
    selection = pd.read_csv(report_dir / "window_model_selection.csv")
    metrics_table = pd.read_csv(metrics_csv)
    with (diagnostics_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics_json = json.load(handle)

    net_metrics = pd.DataFrame([metrics_json.get("net", {})]).T.reset_index()
    net_metrics.columns = ["metric", "value"]
    gross_metrics = pd.DataFrame([metrics_json.get("gross", {})]).T.reset_index()
    gross_metrics.columns = ["metric", "value"]

    with PdfPages(output) as pdf:
        _text_page(
            pdf,
            "DeePM-GAT K=10 Technical Audit",
            [
                "Data: Yahoo public proxy snapshot data_20260625.parquet.",
                f"Model: {run} / {arch}; evaluation path: DeepMomentum seed ensemble.",
                f"Ensemble: top {top_n} validation-ranked seed checkpoints per rolling test window.",
                "Windows: 2010-2014, 2015-2019, and 2020 through available 2026 data.",
                "Caveat: These are diagnostic Yahoo proxy results, not paper-equivalent proprietary futures results.",
            ],
        )
        _table_page(pdf, "Training Artifact Selection Summary", selection_summary)
        _selection_plot_page(pdf, selection)
        _table_page(pdf, "Selected Seed Metrics", selection)
        _table_page(pdf, "Main Metrics Comparison", metrics_table)
        _table_page(pdf, "DeePM Net Metrics", net_metrics)
        _table_page(pdf, "DeePM Gross Metrics", gross_metrics)
        _image_page(pdf, "Cumulative Wealth", diagnostics_dir / "pnl_cumulative.png")
        _drawdown_page(pdf, diagnostics_dir / "pnl_curves.csv")
        _image_page(pdf, "Annual Sharpe", diagnostics_dir / "annual_sharpe.png")
        _image_page(pdf, "Group Gross PnL", diagnostics_dir / "group_gross_pnl.png")
        _image_page(pdf, "Group Net PnL", diagnostics_dir / "group_net_pnl.png")
        _text_page(
            pdf,
            "Interpretation Checklist",
            [
                "Capability: compare net Sharpe, Calmar, drawdown, turnover, and passive benchmark metrics.",
                "Robustness: inspect annual Sharpe and drawdown pages for concentrated regime dependence.",
                "Diversification: compare group gross/net PnL pages for asset-class concentration.",
                "Selection quality: review whether top validation seeds also show stable test metrics.",
                "Costs: use net-vs-gross spread and breakeven cost metrics to judge transaction-cost sensitivity.",
            ],
        )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeePM PDF report")
    parser.add_argument("--model-backtest", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--run", default="deepm-gat")
    parser.add_argument("--arch", default="DeePM")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_report(
        model_backtest=args.model_backtest,
        metrics_csv=Path(args.metrics_csv),
        run=args.run,
        arch=args.arch,
        top_n=args.top_n,
        output=Path(args.output),
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
