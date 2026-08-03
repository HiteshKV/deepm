#!/usr/bin/env python
"""Generate a DeePM/DeRegiME ablation PDF from existing diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_BACKTESTS = [
    "bt-deepm-gat",
    "bt-deepm-gat-dg-mean",
    "bt-deepm-gat-dg-risk",
    "bt-deepm-gat-dg-full",
]

DISPLAY_NAMES = {
    "bt-deepm-gat": "Baseline DeePM-GAT",
    "bt-deepm-gat-dg-mean": "DG Mean",
    "bt-deepm-gat-dg-risk": "DG Risk",
    "bt-deepm-gat-dg-full": "DG Full",
}


def load_metrics(results_dir: Path, name: str) -> dict | None:
    path = results_dir / name / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_frame(results_dir: Path, backtests: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    for name in backtests:
        metrics = load_metrics(results_dir, name)
        if metrics is None:
            missing.append(name)
            continue
        net = metrics.get("net", {})
        gross = metrics.get("gross", {})
        rows.append(
            {
                "Backtest": name,
                "Model": DISPLAY_NAMES.get(name, name),
                "Net Sharpe": net.get("Sharpe Ratio"),
                "Gross Sharpe": gross.get("Sharpe Ratio"),
                "Calmar": net.get("Calmar Ratio"),
                "CAGR %": _pct(net.get("CAGR")),
                "MaxDD %": _pct(net.get("Max Drawdown")),
                "Turnover": net.get("Annualized Turnover (Scaled)"),
                "Net/Gross Gap": (
                    gross.get("Sharpe Ratio") - net.get("Sharpe Ratio")
                    if gross.get("Sharpe Ratio") is not None
                    and net.get("Sharpe Ratio") is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows), missing


def _pct(value):
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 2.0 else value


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.94, title, fontsize=18, fontweight="bold")
    fig.text(0.05, 0.88, "\n".join(lines), fontsize=10, va="top", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    display = frame.copy()
    for col in display.columns:
        if col not in {"Backtest", "Model"}:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_bar_page(pdf: PdfPages, frame: pd.DataFrame, metric: str) -> None:
    if frame.empty or metric not in frame:
        return
    plot = frame.dropna(subset=[metric])
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.bar(plot["Model"], plot[metric])
    ax.set_title(metric, fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def acceptance_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "bt-deepm-gat" not in frame["Backtest"].values:
        return ["Baseline metrics are missing; acceptance cannot be evaluated."]

    base = frame[frame["Backtest"] == "bt-deepm-gat"].iloc[0]
    lines = [
        "Acceptance rule:",
        "  - Net Sharpe improves by >= 0.05, or Calmar improves by >= 5%.",
        "  - Max drawdown and turnover must not worsen by more than 10%.",
        "",
    ]
    for _, row in frame[frame["Backtest"] != "bt-deepm-gat"].iterrows():
        sharpe_ok = row["Net Sharpe"] >= base["Net Sharpe"] + 0.05
        calmar_ok = row["Calmar"] >= base["Calmar"] * 1.05
        drawdown_ok = abs(row["MaxDD %"]) <= abs(base["MaxDD %"]) * 1.10
        turnover_ok = row["Turnover"] <= base["Turnover"] * 1.10
        accepted = (sharpe_ok or calmar_ok) and drawdown_ok and turnover_ok
        lines.append(
            f"{row['Model']}: {'PASS' if accepted else 'HOLD'} | "
            f"Sharpe {row['Net Sharpe']:.3f} vs {base['Net Sharpe']:.3f}; "
            f"Calmar {row['Calmar']:.3f} vs {base['Calmar']:.3f}; "
            f"MaxDD {row['MaxDD %']:.2f}% vs {base['MaxDD %']:.2f}%; "
            f"Turnover {row['Turnover']:.2f} vs {base['Turnover']:.2f}"
        )
    return lines


def audit_lines(audit_path: Path) -> list[str]:
    if not audit_path.exists():
        return [f"Sidecar audit not found: {audit_path}"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validation = audit.get("validation", {})
    return [
        f"Audit: {audit_path}",
        f"Generated: {audit.get('generated_at_utc')}",
        f"Method: {audit.get('method')}",
        f"Target column: {audit.get('target_col')}",
        f"Target lag: {audit.get('target_lag')}",
        f"Horizons: {audit.get('horizons')}",
        f"Sidecar rows: {validation.get('sidecar_rows')}",
        f"Tickers: {validation.get('sidecar_tickers')}",
        f"Date range: {validation.get('sidecar_first_date')} to {validation.get('sidecar_last_date')}",
        "",
        audit.get("leakage_policy", ""),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeRegiME ablation report")
    parser.add_argument("--results-dir", type=Path, default=Path("backtest_diagnostics"))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/deregime/deregime_sidecar_audit_20260625.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/deregime_deepm_ablation_report.pdf"),
    )
    parser.add_argument("backtests", nargs="*", default=DEFAULT_BACKTESTS)
    args = parser.parse_args()

    frame, missing = metrics_frame(args.results_dir, args.backtests)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.output) as pdf:
        add_text_page(
            pdf,
            "DeRegiME-DeePM Ablation Report",
            [
                "This report compares DeePM-GAT against optional DeRegiME sidecar feature variants.",
                "The sidecar is diagnostic on Yahoo proxy data and is not paper-equivalent.",
                "",
                *audit_lines(args.audit),
                "",
                "Missing diagnostics:",
                *(missing or ["None"]),
            ],
        )
        if not frame.empty:
            add_table_page(pdf, "Trading Metrics", frame)
            for metric in ["Net Sharpe", "Calmar", "MaxDD %", "Turnover", "Net/Gross Gap"]:
                add_bar_page(pdf, frame, metric)
        add_text_page(pdf, "Acceptance Check", acceptance_lines(frame))

    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
