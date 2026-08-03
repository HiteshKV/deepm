"""Daily report rendering and email delivery."""

from __future__ import annotations

import html
import json
import os
import shlex
import smtplib
import ssl
import urllib.error
import urllib.request
from dataclasses import fields
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

import pandas as pd

from deepm._paths import PROJECT_ROOT
from deepm.live.exceptions import LiveTradingError
from deepm.live.types import (
    Fill,
    Instrument,
    OrderIntent,
    Position,
    RiskCheck,
    Signal,
    TargetPosition,
    dataclass_to_dict,
)


def _frame(values: Iterable[object]) -> pd.DataFrame:
    rows = [dataclass_to_dict(value) for value in values]
    return pd.DataFrame(rows)


def _columns_for(cls: type[object]) -> list[str]:
    return [field.name for field in fields(cls)]


def write_csv(path: Path, values: Iterable[object], columns: list[str] | None = None) -> pd.DataFrame:
    frame = _frame(values)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)
    return frame


def _money(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.2f}"


def _currency_money(value: object, currency: str | None = None) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    currency = (currency or "USD").upper()
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    sign = "-" if number < 0 else ""
    symbol = symbols.get(currency)
    if symbol:
        return f"{sign}{symbol}{abs(number):,.2f}"
    return f"{sign}{abs(number):,.2f} {currency}"


def _pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number * 100:.2f}%"


def _num(value: object, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.{decimals}f}"
    return f"{number:.{decimals}f}"


def _mode_label(mode: str) -> str:
    return {
        "sim": "Local paper simulator",
        "ibkr-paper": "IBKR paper simulation",
        "ibkr-live": "IBKR live trading",
    }.get(mode, mode)


def _status_label(status: str) -> str:
    return {
        "completed": "Completed",
        "already_ran": "Already ran",
        "dry_run": "Dry run",
        "failed": "Failed",
    }.get(status, status.replace("_", " ").title())


def _instrument_name(ticker: str, instruments: dict[str, Instrument] | None) -> str:
    instrument = (instruments or {}).get(str(ticker))
    if instrument is None:
        return str(ticker)
    return instrument.description or instrument.bloomberg_ticker or instrument.ticker


def _contract_label(ticker: str, instruments: dict[str, Instrument] | None) -> str:
    instrument = (instruments or {}).get(str(ticker))
    if instrument is None:
        return str(ticker)
    return f"{instrument.ibkr_symbol} {instrument.sec_type} on {instrument.exchange}"


def _instrument_currency(ticker: str, instruments: dict[str, Instrument] | None) -> str:
    instrument = (instruments or {}).get(str(ticker))
    return instrument.currency if instrument is not None else "USD"


def _side_label(side: str) -> str:
    side = str(side).upper()
    if side == "BUY":
        return "Buy"
    if side == "SELL":
        return "Sell"
    return side.title()


def _direction(quantity: int) -> str:
    if quantity > 0:
        return "Long"
    if quantity < 0:
        return "Short"
    return "Flat"


def _signal_label(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number > 0.15:
        return "Long bias"
    if number < -0.15:
        return "Short bias"
    return "Near neutral"


def _section(title: str, body: str, klass: str = "") -> str:
    return f'<section class="panel {klass}"><h2>{html.escape(title)}</h2>{body}</section>'


def _empty(message: str = "Nothing to report.") -> str:
    return f'<p class="muted">{html.escape(message)}</p>'


def _html_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return _empty("None.")
    header = "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key, _label in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _orders_rows(
    orders: list[OrderIntent],
    instruments: dict[str, Instrument] | None,
) -> list[dict[str, object]]:
    rows = []
    for order in orders:
        rows.append(
            {
                "instrument": _instrument_name(order.ticker, instruments),
                "contract": _contract_label(order.ticker, instruments),
                "action": _side_label(order.side),
                "quantity": order.quantity,
                "from_to": f"{order.current_quantity} -> {order.target_quantity}",
                "notional": _currency_money(
                    order.quantity * order.contract_value,
                    order.currency,
                ),
                "price": _num(order.limit_price, 5),
                "reason": str(order.reason).replace("_", " "),
            }
        )
    return rows


def _fills_rows(
    fills: list[Fill],
    instruments: dict[str, Instrument] | None,
) -> list[dict[str, object]]:
    rows = []
    for fill in fills:
        rows.append(
            {
                "instrument": _instrument_name(fill.ticker, instruments),
                "action": _side_label(fill.side),
                "quantity": fill.quantity,
                "price": _num(fill.price, 5),
                "notional": _currency_money(
                    fill.quantity * fill.price * fill.point_value,
                    _instrument_currency(fill.ticker, instruments),
                ),
                "status": str(fill.status).title(),
                "commission": _money(fill.commission),
            }
        )
    return rows


def _positions_rows(
    positions: list[Position],
    instruments: dict[str, Instrument] | None,
) -> list[dict[str, object]]:
    rows = []
    for position in sorted(
        positions,
        key=lambda item: abs(item.market_value),
        reverse=True,
    ):
        rows.append(
            {
                "instrument": _instrument_name(position.ticker, instruments),
                "direction": _direction(position.quantity),
                "quantity": abs(position.quantity),
                "market_price": _num(position.market_price, 5),
                "market_value": _currency_money(
                    position.market_value,
                    _instrument_currency(position.ticker, instruments),
                ),
            }
        )
    return rows


def _signals_rows(
    signals: list[Signal],
    instruments: dict[str, Instrument] | None,
    limit: int = 12,
) -> list[dict[str, object]]:
    ranked = sorted(signals, key=lambda item: abs(item.model_position), reverse=True)
    rows = []
    for signal in ranked[:limit]:
        rows.append(
            {
                "instrument": _instrument_name(signal.ticker, instruments),
                "bias": _signal_label(signal.model_position),
                "model_position": _pct(signal.model_position),
            }
        )
    return rows


def _plain_english_notes(
    status: str,
    summary: dict[str, object],
    orders: list[OrderIntent],
    fills: list[Fill],
    errors: list[str],
) -> list[str]:
    notes = []
    if status == "completed":
        notes.append("The daily run completed and the paper account ledger was updated.")
    elif status == "already_ran":
        notes.append("This date had already been processed, so no duplicate trades were created.")
    elif status == "dry_run":
        notes.append("The model produced orders, but order placement was disabled for this run.")
    else:
        notes.append("The run did not complete. Review the detailed report before taking action.")

    if orders:
        notes.append(f"The model requested {len(orders)} order(s); {len(fills)} fill(s) were recorded.")
    else:
        notes.append("The model did not request any new trades today.")

    daily_pnl = summary.get("daily_pnl")
    if daily_pnl is not None:
        notes.append(f"Estimated daily P/L was {_money(daily_pnl)} before any live-broker reconciliation.")

    if errors:
        notes.append(f"There were {len(errors)} error message(s); they are shown in the detailed report.")
    return notes


def render_html_report(
    run_date: date,
    mode: str,
    status: str,
    summary: dict[str, object],
    signals: list[Signal],
    targets: list[TargetPosition],
    orders: list[OrderIntent],
    fills: list[Fill],
    checks: list[RiskCheck],
    errors: list[str],
    positions: list[Position] | None = None,
    report_title: str = "DeePM Daily Report",
    instruments: dict[str, Instrument] | None = None,
) -> str:
    """Render a full technical HTML report for a daily run."""
    positions = positions or []

    def table(title: str, values: Iterable[object]) -> str:
        frame = _frame(values)
        if not frame.empty and "ticker" in frame.columns:
            frame.insert(
                frame.columns.get_loc("ticker") + 1,
                "instrument",
                frame["ticker"].map(lambda ticker: _instrument_name(ticker, instruments)),
            )
        if frame.empty:
            return f"<h2>{html.escape(title)}</h2><p>None.</p>"
        return f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, escape=True)}"

    checks_frame = _frame(checks)
    failures = checks_frame[checks_frame["passed"] == False] if not checks_frame.empty else pd.DataFrame()  # noqa: E712
    error_html = "".join(f"<li>{html.escape(error)}</li>" for error in errors)
    summary_html = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(report_title)} {html.escape(run_date.isoformat())}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; background: #f6f8fb; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 22px; background: white; }}
    th, td {{ border-bottom: 1px solid #e6ebf1; padding: 7px 9px; font-size: 12px; text-align: right; }}
    th {{ background: #eef3f8; text-align: left; color: #314054; }}
    h1, h2 {{ margin-bottom: 8px; }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 18px; margin-top: 26px; }}
    .hero {{ background: #0f2f4a; color: white; padding: 22px 24px; border-radius: 10px; }}
    .status {{ font-weight: 700; }}
    .failed {{ color: #b00020; }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>{html.escape(report_title)}</h1>
    <p>Date: <strong>{html.escape(run_date.isoformat())}</strong> | Mode: <strong>{html.escape(_mode_label(mode))}</strong> | Status: <span class="status">{html.escape(_status_label(status))}</span></p>
  </div>
  <h2>Summary</h2>
  <table>{summary_html}</table>
  <p class="failed">Failed checks: {len(failures)}</p>
  <ul>{error_html}</ul>
  {table("Orders", orders)}
  {table("Fills", fills)}
  {table("Current Positions", positions)}
  {table("Risk Checks", checks)}
  {table("Targets", targets)}
  {table("Signals", signals)}
</div>
</body>
</html>
"""


def render_legacy_html_report(
    run_date: date,
    mode: str,
    status: str,
    summary: dict[str, object],
    signals: list[Signal],
    targets: list[TargetPosition],
    orders: list[OrderIntent],
    fills: list[Fill],
    checks: list[RiskCheck],
    errors: list[str],
    positions: list[Position] | None = None,
    report_title: str = "DeePM Daily Report",
) -> str:
    """Render the original compact technical report layout."""
    positions = positions or []

    def table(title: str, values: Iterable[object]) -> str:
        frame = _frame(values)
        if frame.empty:
            return f"<h2>{html.escape(title)}</h2><p>None.</p>"
        return f"<h2>{html.escape(title)}</h2>{frame.to_html(index=False, escape=True)}"

    checks_frame = _frame(checks)
    failures = checks_frame[checks_frame["passed"] == False] if not checks_frame.empty else pd.DataFrame()  # noqa: E712
    error_html = "".join(f"<li>{html.escape(error)}</li>" for error in errors)
    summary_html = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(report_title)} {html.escape(run_date.isoformat())}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align: right; }}
    th {{ background: #f4f4f4; text-align: left; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .status {{ font-weight: 700; }}
    .failed {{ color: #b00020; }}
  </style>
</head>
<body>
  <h1>{html.escape(report_title)}</h1>
  <p>Date: <strong>{html.escape(run_date.isoformat())}</strong> | Mode: <strong>{html.escape(mode)}</strong> | Status: <span class="status">{html.escape(status)}</span></p>
  <h2>Summary</h2>
  <table>{summary_html}</table>
  <p class="failed">Failed checks: {len(failures)}</p>
  <ul>{error_html}</ul>
  {table("Orders", orders)}
  {table("Fills", fills)}
  {table("Current Positions", positions)}
  {table("Risk Checks", checks)}
  {table("Targets", targets)}
  {table("Signals", signals)}
</body>
</html>
"""


def render_friendly_html_report(
    run_date: date,
    mode: str,
    status: str,
    summary: dict[str, object],
    signals: list[Signal],
    orders: list[OrderIntent],
    fills: list[Fill],
    errors: list[str],
    positions: list[Position] | None = None,
    instruments: dict[str, Instrument] | None = None,
    report_title: str = "DeePM Daily Trade Summary",
) -> str:
    """Render a human-friendly daily report for email."""
    positions = positions or []
    status_class = "good" if status in {"completed", "already_ran", "dry_run"} and not errors else "bad"
    cards = [
        ("Account value", _money(summary.get("ending_equity", summary.get("account_equity")))),
        ("Daily P/L", _money(summary.get("daily_pnl"))),
        ("Daily return", _pct(summary.get("daily_return"))),
        ("Current exposure", _money(summary.get("gross_exposure"))),
        ("Open positions", str(summary.get("position_count", len(positions)))),
        ("Trades filled", str(summary.get("fills", len(fills)))),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    notes = _plain_english_notes(status, summary, orders, fills, errors)
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    error_html = "".join(f"<li>{html.escape(error)}</li>" for error in errors)

    orders_table = _html_table(
        _orders_rows(orders, instruments),
        [
            ("instrument", "What"),
            ("action", "Action"),
            ("quantity", "Contracts"),
            ("from_to", "Position change"),
            ("notional", "Approx. notional"),
            ("price", "Limit price"),
        ],
    )
    fills_table = _html_table(
        _fills_rows(fills, instruments),
        [
            ("instrument", "What"),
            ("action", "Action"),
            ("quantity", "Contracts"),
            ("price", "Fill price"),
            ("notional", "Approx. notional"),
            ("status", "Status"),
        ],
    )
    positions_table = _html_table(
        _positions_rows(positions, instruments),
        [
            ("instrument", "What"),
            ("direction", "Direction"),
            ("quantity", "Contracts"),
            ("market_price", "Market price"),
            ("market_value", "Market value"),
        ],
    )
    signals_table = _html_table(
        _signals_rows(signals, instruments),
        [
            ("instrument", "What"),
            ("bias", "Model view"),
            ("model_position", "Signal strength"),
        ],
    )

    if not orders:
        headline = "No new trades today"
    elif len(orders) == 1:
        headline = "1 trade requested today"
    else:
        headline = f"{len(orders)} trades requested today"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(report_title)} {html.escape(run_date.isoformat())}</title>
  <style>
    body {{ margin: 0; background: #f5f7fa; color: #17202a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 28px 18px 40px; }}
    .hero {{ background: #12324a; color: white; border-radius: 14px; padding: 26px 28px; }}
    .eyebrow {{ font-size: 13px; text-transform: uppercase; letter-spacing: .06em; opacity: .78; margin: 0 0 8px; }}
    h1 {{ margin: 0; font-size: 30px; line-height: 1.15; }}
    .sub {{ margin: 12px 0 0; color: #d9e7f2; }}
    .badge {{ display: inline-block; margin-top: 16px; padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; }}
    .badge.good {{ background: #d9f8e5; color: #165b31; }}
    .badge.bad {{ background: #ffe0df; color: #8a1f17; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #e6ebf1; border-radius: 12px; padding: 15px; }}
    .card span {{ display: block; color: #667085; font-size: 13px; margin-bottom: 7px; }}
    .card strong {{ font-size: 21px; }}
    .panel {{ background: white; border: 1px solid #e6ebf1; border-radius: 12px; padding: 18px; margin: 16px 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #edf1f5; padding: 10px 8px; text-align: left; font-size: 14px; vertical-align: top; }}
    th {{ color: #475467; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .muted {{ color: #667085; }}
    .errors {{ color: #9f1d1d; }}
    ul {{ padding-left: 20px; margin: 8px 0 0; }}
    li {{ margin-bottom: 7px; }}
    @media (max-width: 720px) {{ .cards {{ grid-template-columns: 1fr; }} .wrap {{ padding: 14px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <p class="eyebrow">{html.escape(_mode_label(mode))} · {html.escape(run_date.isoformat())}</p>
      <h1>{html.escape(headline)}</h1>
      <p class="sub">This is the daily paper-trading summary for the DeePM model. It shows what the model changed, what filled, and how the paper account moved.</p>
      <span class="badge {status_class}">{html.escape(_status_label(status))}</span>
    </div>
    <div class="cards">{card_html}</div>
    {_section("What happened today", f"<ul>{notes_html}</ul>")}
    {_section("Orders requested", orders_table)}
    {_section("Fills recorded", fills_table)}
    {_section("Current paper positions", positions_table)}
    {_section("Errors", f'<ul class="errors">{error_html}</ul>' if errors else _empty("No errors were reported."))}
    {_section("Top model signals", signals_table)}
    <p class="muted">A full audit report with raw targets, signals, and risk checks is saved as <strong>report_detailed.html</strong> in the same daily run folder.</p>
  </div>
</body>
</html>
"""


def write_run_outputs(
    run_dir: Path,
    run_date: date,
    mode: str,
    status: str,
    summary: dict[str, object],
    input_snapshot: dict[str, object],
    signals: list[Signal],
    targets: list[TargetPosition],
    orders: list[OrderIntent],
    fills: list[Fill],
    checks: list[RiskCheck],
    errors: list[str],
    positions: list[Position] | None = None,
    instruments: dict[str, Instrument] | None = None,
) -> Path:
    """Write the immutable daily audit files."""
    positions = positions or []
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_snapshot.json").write_text(
        json.dumps(input_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(run_dir / "signals.csv", signals, _columns_for(Signal))
    write_csv(run_dir / "targets.csv", targets, _columns_for(TargetPosition))
    write_csv(run_dir / "orders.csv", orders, _columns_for(OrderIntent))
    write_csv(run_dir / "fills.csv", fills, _columns_for(Fill))
    risk_json = [dataclass_to_dict(check) for check in checks]
    (run_dir / "risk_checks.json").write_text(
        json.dumps(risk_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(run_dir / "positions.csv", positions, _columns_for(Position))
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    status_payload = {"date": run_date.isoformat(), "mode": mode, "status": status, "errors": errors}
    (run_dir / "status.json").write_text(
        json.dumps(status_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    detailed_report = render_html_report(
        run_date,
        mode,
        status,
        summary,
        signals,
        targets,
        orders,
        fills,
        checks,
        errors,
        positions,
        report_title="DeePM Detailed Daily Audit",
        instruments=instruments,
    )
    detailed_report_path = run_dir / "report_detailed.html"
    detailed_report_path.write_text(detailed_report, encoding="utf-8")
    legacy_report = render_legacy_html_report(
        run_date,
        mode,
        status,
        summary,
        signals,
        targets,
        orders,
        fills,
        checks,
        errors,
        positions,
    )
    (run_dir / "report_legacy.html").write_text(legacy_report, encoding="utf-8")
    report = render_friendly_html_report(
        run_date,
        mode,
        status,
        summary,
        signals,
        orders,
        fills,
        errors,
        positions,
        instruments=instruments,
    )
    report_path = run_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _read_csv_dataclasses(path: Path, cls: type[object]) -> list[object]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    allowed = {field.name for field in fields(cls)}
    rows = []
    for raw in frame.to_dict(orient="records"):
        data = {key: value for key, value in raw.items() if key in allowed and not pd.isna(value)}
        for key in ("date", "run_date"):
            if key in data:
                data[key] = date.fromisoformat(str(data[key]))
        rows.append(cls(**data))
    return rows


def _read_risk_checks(path: Path) -> list[RiskCheck]:
    if not path.exists():
        return []
    values = json.loads(path.read_text(encoding="utf-8"))
    return [RiskCheck(**value) for value in values]


def regenerate_reports_from_run_dir(
    run_dir: Path,
    run_date: date,
    mode: str,
    instruments: dict[str, Instrument] | None = None,
) -> Path:
    """Re-render friendly and detailed reports from existing audit files."""
    summary_path = run_dir / "summary.json"
    status_path = run_dir / "status.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {"date": run_date.isoformat(), "mode": mode}
    )
    status_payload = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    status = str(status_payload.get("status", "unknown"))
    errors = list(status_payload.get("errors", []))
    signals = _read_csv_dataclasses(run_dir / "signals.csv", Signal)
    targets = _read_csv_dataclasses(run_dir / "targets.csv", TargetPosition)
    orders = _read_csv_dataclasses(run_dir / "orders.csv", OrderIntent)
    fills = _read_csv_dataclasses(run_dir / "fills.csv", Fill)
    positions = _read_csv_dataclasses(run_dir / "positions.csv", Position)
    checks = _read_risk_checks(run_dir / "risk_checks.json")

    detailed_report = render_html_report(
        run_date,
        mode,
        status,
        summary,
        signals,
        targets,
        orders,
        fills,
        checks,
        errors,
        positions,
        report_title="DeePM Detailed Daily Audit",
        instruments=instruments,
    )
    (run_dir / "report_detailed.html").write_text(detailed_report, encoding="utf-8")
    legacy_report = render_legacy_html_report(
        run_date,
        mode,
        status,
        summary,
        signals,
        targets,
        orders,
        fills,
        checks,
        errors,
        positions,
    )
    (run_dir / "report_legacy.html").write_text(legacy_report, encoding="utf-8")
    friendly_report = render_friendly_html_report(
        run_date,
        mode,
        status,
        summary,
        signals,
        orders,
        fills,
        errors,
        positions,
        instruments=instruments,
    )
    report_path = run_dir / "report.html"
    report_path.write_text(friendly_report, encoding="utf-8")
    return report_path


def _recipients(email_config: dict[str, object]) -> list[str]:
    recipients = list(email_config.get("to", []))
    if not recipients:
        raise LiveTradingError("No report recipients configured")
    return recipients


def _email_subject(
    email_config: dict[str, object],
    run_date: date,
    mode: str,
    report_label: str | None = None,
) -> str:
    prefix = str(email_config.get("subject_prefix", "[DeePM Live]"))
    suffix = f" - {report_label}" if report_label else ""
    return f"{prefix} {mode} {run_date.isoformat()}{suffix}"


def load_local_live_env() -> None:
    """Load .env.live for direct Python CLI runs without overwriting shell env."""
    if os.getenv("DEEPM_DISABLE_DOTENV") == "1":
        return
    path = PROJECT_ROOT / ".env.live"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed = shlex.split(value)
        except ValueError:
            parsed = []
        if parsed:
            value = parsed[0]
        else:
            value = value.strip("'\"")
        os.environ.setdefault(key, value)


def _send_smtp_report(
    report_path: Path,
    email_config: dict[str, object],
    run_date: date,
    mode: str,
    report_label: str | None = None,
) -> None:
    """Send the HTML report using SMTP environment variables."""
    required = [
        "DEEPM_SMTP_HOST",
        "DEEPM_SMTP_PORT",
        "DEEPM_SMTP_USER",
        "DEEPM_SMTP_PASSWORD",
        "DEEPM_EMAIL_FROM",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise LiveTradingError(f"Missing SMTP environment variables: {', '.join(missing)}")

    recipients = _recipients(email_config)
    message = EmailMessage()
    message["Subject"] = _email_subject(email_config, run_date, mode, report_label)
    message["From"] = os.environ["DEEPM_EMAIL_FROM"]
    message["To"] = ", ".join(recipients)
    html_body = report_path.read_text(encoding="utf-8")
    message.set_content("DeePM daily report is included below as HTML.")
    message.add_alternative(html_body, subtype="html")

    host = os.environ["DEEPM_SMTP_HOST"]
    port = int(os.environ["DEEPM_SMTP_PORT"])
    user = os.environ["DEEPM_SMTP_USER"]
    password = os.environ["DEEPM_SMTP_PASSWORD"]
    use_ssl = os.getenv("DEEPM_SMTP_USE_SSL", "0") == "1" or port == 465
    if use_ssl:
        smtp_context = smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context())
    else:
        smtp_context = smtplib.SMTP(host, port, timeout=30)

    with smtp_context as smtp:
        if not use_ssl:
            smtp.starttls(context=ssl.create_default_context())
        smtp.login(user, password)
        smtp.send_message(message)


def _send_resend_report(
    report_path: Path,
    email_config: dict[str, object],
    run_date: date,
    mode: str,
    report_label: str | None = None,
) -> None:
    """Send the HTML report using Resend's HTTPS email API."""
    api_key = os.getenv("DEEPM_RESEND_API_KEY") or os.getenv("RESEND_API_KEY")
    if not api_key:
        raise LiveTradingError("Missing DEEPM_RESEND_API_KEY or RESEND_API_KEY")

    recipients = _recipients(email_config)
    sender = os.getenv("DEEPM_RESEND_FROM") or os.getenv(
        "DEEPM_EMAIL_FROM",
        "DeePM Reports <onboarding@resend.dev>",
    )
    payload = {
        "from": sender,
        "to": recipients,
        "subject": _email_subject(email_config, run_date, mode, report_label),
        "html": report_path.read_text(encoding="utf-8"),
        "text": f"DeePM daily report: {mode} {run_date.isoformat()} {report_label or ''}".strip(),
    }
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "deepm-live/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise LiveTradingError(f"Resend API returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LiveTradingError(f"Resend API error HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LiveTradingError(f"Resend API connection failed: {exc}") from exc


def send_email_report(
    report_path: Path,
    email_config: dict[str, object],
    run_date: date,
    mode: str,
    report_label: str | None = None,
) -> None:
    """Send the HTML report through the configured email provider."""
    if not report_path.exists():
        raise LiveTradingError(f"Report not found: {report_path}")
    load_local_live_env()
    provider = os.getenv("DEEPM_EMAIL_PROVIDER", "smtp").strip().lower()
    if provider == "resend":
        _send_resend_report(report_path, email_config, run_date, mode, report_label)
        return
    if provider == "smtp":
        _send_smtp_report(report_path, email_config, run_date, mode, report_label)
        return
    raise LiveTradingError(f"Unknown DEEPM_EMAIL_PROVIDER: {provider}")


def send_daily_report_emails(
    run_dir: Path,
    email_config: dict[str, object],
    run_date: date,
    mode: str,
    include_detailed: bool | None = None,
    include_legacy: bool | None = None,
) -> list[Path]:
    """Send one or more daily report emails with explicit report labels."""
    if include_detailed is None:
        include_detailed = bool(email_config.get("send_detailed", False))
    if include_legacy is None:
        include_legacy = bool(email_config.get("send_legacy", False))

    reports: list[tuple[Path, str]] = [
        (run_dir / "report.html", "Friendly Summary"),
    ]
    if include_legacy:
        reports.append((run_dir / "report_legacy.html", "Legacy Technical Report"))
    if include_detailed:
        reports.append((run_dir / "report_detailed.html", "Detailed Audit"))

    sent: list[Path] = []
    for path, label in reports:
        send_email_report(path, email_config, run_date, mode, report_label=label)
        sent.append(path)
    return sent
