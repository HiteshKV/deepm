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
) -> str:
    """Render a compact HTML report for a daily run."""
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
    report = render_html_report(
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
    report_path = run_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _recipients(email_config: dict[str, object]) -> list[str]:
    recipients = list(email_config.get("to", []))
    if not recipients:
        raise LiveTradingError("No report recipients configured")
    return recipients


def _email_subject(email_config: dict[str, object], run_date: date, mode: str) -> str:
    prefix = str(email_config.get("subject_prefix", "[DeePM Live]"))
    return f"{prefix} {mode} {run_date.isoformat()}"


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
    message["Subject"] = _email_subject(email_config, run_date, mode)
    message["From"] = os.environ["DEEPM_EMAIL_FROM"]
    message["To"] = ", ".join(recipients)
    html_body = report_path.read_text(encoding="utf-8")
    message.set_content("DeePM daily report is attached as HTML.")
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
        "subject": _email_subject(email_config, run_date, mode),
        "html": report_path.read_text(encoding="utf-8"),
        "text": f"DeePM daily report: {mode} {run_date.isoformat()}",
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


def send_email_report(report_path: Path, email_config: dict[str, object], run_date: date, mode: str) -> None:
    """Send the HTML report through the configured email provider."""
    load_local_live_env()
    provider = os.getenv("DEEPM_EMAIL_PROVIDER", "smtp").strip().lower()
    if provider == "resend":
        _send_resend_report(report_path, email_config, run_date, mode)
        return
    if provider == "smtp":
        _send_smtp_report(report_path, email_config, run_date, mode)
        return
    raise LiveTradingError(f"Unknown DEEPM_EMAIL_PROVIDER: {provider}")
