"""Inspect or email an existing daily live report."""

from __future__ import annotations

import argparse
from datetime import date

from deepm.live.config import DEFAULT_CONFIG, load_live_config, path_from_config
from deepm.live.instruments import load_instruments
from deepm.live.reporting import (
    regenerate_reports_from_run_dir,
    send_daily_report_emails,
    send_email_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open or send an existing DeePM live report")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", default="sim")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--send-detailed", action="store_true")
    parser.add_argument("--send-legacy", action="store_true")
    parser.add_argument("--send-both", action="store_true")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date)
    config = load_live_config(args.config)
    run_dir = path_from_config(config, "paths", "run_dir") / run_date.isoformat()
    report_path = run_dir / "report.html"
    if args.regenerate:
        instruments = load_instruments(path_from_config(config, "paths", "instrument_map"))
        report_path = regenerate_reports_from_run_dir(
            run_dir,
            run_date,
            args.mode,
            instruments=instruments,
        )
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")
    email_config = config.get("email", {})
    if args.send_both:
        send_daily_report_emails(
            run_dir,
            email_config,
            run_date,
            args.mode,
            include_detailed=True,
        )
    else:
        if args.send_email:
            send_email_report(
                report_path,
                email_config,
                run_date,
                args.mode,
                report_label="Friendly Summary",
            )
        if args.send_detailed:
            send_email_report(
                run_dir / "report_detailed.html",
                email_config,
                run_date,
                args.mode,
                report_label="Detailed Audit",
            )
        if args.send_legacy:
            send_email_report(
                run_dir / "report_legacy.html",
                email_config,
                run_date,
                args.mode,
                report_label="Legacy Technical Report",
            )
    print(report_path)


if __name__ == "__main__":
    main()
