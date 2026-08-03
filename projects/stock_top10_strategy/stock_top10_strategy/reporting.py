"""HTML report generation for top-10 stock strategy runs."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_report(run_dir: Path) -> Path:
    """Build charts and an HTML report for a completed run."""
    nav = pd.read_csv(run_dir / "nav.csv")
    fills = _read_optional_csv(run_dir / "fills.csv")
    holdings = _read_optional_csv(run_dir / "holdings.csv")
    membership = _read_optional_csv(run_dir / "membership.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    nav["date"] = pd.to_datetime(nav["date"])
    nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
    chart_path = run_dir / "nav_drawdown.png"
    _write_nav_chart(nav, chart_path)

    report = render_report(nav, fills, holdings, membership, metrics, chart_path.name)
    output = run_dir / "report.html"
    output.write_text(report, encoding="utf-8")
    return output


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_nav_chart(nav: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(nav["date"], nav["nav"], color="#1f5f8b")
    axes[0].set_title("Portfolio NAV")
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(nav["date"], nav["drawdown"], 0, color="#b83232", alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_report(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    holdings: pd.DataFrame,
    membership: pd.DataFrame,
    metrics: dict[str, object],
    chart_name: str,
) -> str:
    """Render a self-contained HTML report shell."""
    latest_holdings = pd.DataFrame()
    if not holdings.empty:
        latest_date = holdings["date"].max()
        latest_holdings = holdings[holdings["date"] == latest_date].sort_values(
            "market_value_usd",
            ascending=False,
        )

    annual = _annual_returns(nav)
    monthly = _monthly_returns(nav)
    filled = fills[fills.get("status", pd.Series(dtype=str)) == "filled"] if not fills.empty else pd.DataFrame()
    recent_fills = filled.tail(25) if not filled.empty else pd.DataFrame()
    latest_members = pd.DataFrame()
    if not membership.empty:
        latest_members = membership[membership["date"] == membership["date"].max()].sort_values("rank")

    cards = "".join(
        f"<div class='card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in [
            ("Ending NAV", _money(metrics.get("ending_nav"))),
            ("Total return", _pct(metrics.get("total_return"))),
            ("CAGR", _pct(metrics.get("cagr"))),
            ("Sharpe", _num(metrics.get("sharpe"))),
            ("Max drawdown", _pct(metrics.get("max_drawdown"))),
            ("Total costs", _money(metrics.get("total_costs_usd"))),
        ]
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Top-10 Buy-the-Move Backtest</title>
  <style>
    body {{ margin: 0; background: #f6f8fb; color: #17202a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 44px; }}
    .hero {{ background: #14324a; color: white; border-radius: 12px; padding: 24px 28px; }}
    .hero p {{ color: #d9e7f2; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .card, .panel {{ background: white; border: 1px solid #e6ebf1; border-radius: 10px; padding: 16px; }}
    .card span {{ color: #667085; display: block; font-size: 13px; margin-bottom: 6px; }}
    .card strong {{ font-size: 22px; }}
    .panel {{ margin: 16px 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #edf1f5; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ color: #475467; text-transform: uppercase; font-size: 12px; letter-spacing: .04em; }}
    img {{ max-width: 100%; }}
    @media (max-width: 760px) {{ .cards {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Top-10 Global Stocks Buy-the-Move Backtest</h1>
      <p>Point-in-time market-cap membership, fixed-notional buy signals, cash accounting, transaction costs, and gradual exits for names leaving the top 10.</p>
      <p>Data source: {html.escape(str(metrics.get("data_source_label", "unknown")))} ({html.escape(str(metrics.get("data_source_kind", "unknown")))})</p>
    </div>
    <div class="cards">{cards}</div>
    <div class="panel"><img src="{html.escape(chart_name)}" alt="NAV and drawdown chart"></div>
    {_table_section("Latest Top-10 Membership", latest_members)}
    {_table_section("Current Holdings", latest_holdings)}
    {_table_section("Recent Filled Trades", recent_fills)}
    {_table_section("Annual Returns", annual)}
    {_table_section("Monthly Returns", monthly)}
  </div>
</body>
</html>
"""


def _annual_returns(nav: pd.DataFrame) -> pd.DataFrame:
    frame = nav.copy()
    frame["year"] = frame["date"].dt.year
    rows = []
    for year, group in frame.groupby("year"):
        rows.append({"year": year, "return": group["nav"].iloc[-1] / group["nav"].iloc[0] - 1.0})
    return pd.DataFrame(rows)


def _monthly_returns(nav: pd.DataFrame) -> pd.DataFrame:
    frame = nav.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    rows = []
    for month, group in frame.groupby("month"):
        rows.append({"month": month, "return": group["nav"].iloc[-1] / group["nav"].iloc[0] - 1.0})
    return pd.DataFrame(rows)


def _table_section(title: str, frame: pd.DataFrame) -> str:
    if frame.empty:
        body = "<p>None.</p>"
    else:
        body = frame.to_html(index=False, escape=True)
    return f"<div class='panel'><h2>{html.escape(title)}</h2>{body}</div>"


def _money(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"
