"""Simple report generator for hourly equity backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from equity_intraday_deepm.config import project_path


def generate_report(title: str, output: str | Path) -> Path:
    output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_paths = sorted((project_path("equity_intraday_deepm/backtests")).glob("*/metrics.json"))
    rows = []
    for path in metrics_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["backtest"] = path.parent.name
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No hourly equity backtest metrics found")
    metrics = pd.DataFrame(rows)
    html_path = output.with_suffix(".html")
    html_path.write_text(_html(title, metrics), encoding="utf-8")
    if output.suffix.lower() == ".pdf":
        _pdf(title, metrics, output)
        return output
    return html_path


def _html(title: str, metrics: pd.DataFrame) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.4}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f4f4f4}}</style>
</head><body><h1>{title}</h1>
<p>Hourly top-10 global equity DeePM pilot results. Vendor-grade point-in-time data is required for production research.</p>
{metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
</body></html>"""


def _pdf(title: str, metrics: pd.DataFrame, output: Path) -> None:
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=16, pad=20)
        table = ax.table(cellText=metrics.round(4).astype(str).values, colLabels=metrics.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate hourly equity report")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(generate_report(args.title, args.output))


if __name__ == "__main__":
    main()
