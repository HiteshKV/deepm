"""Inspect or email an existing daily live report."""

from __future__ import annotations

import argparse
from datetime import date

from deepm.live.config import DEFAULT_CONFIG, load_live_config, path_from_config
from deepm.live.reporting import send_email_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Open or send an existing DeePM live report")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", default="sim")
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date)
    config = load_live_config(args.config)
    report_path = path_from_config(config, "paths", "run_dir") / run_date.isoformat() / "report.html"
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")
    if args.send_email:
        send_email_report(report_path, config.get("email", {}), run_date, args.mode)
    print(report_path)


if __name__ == "__main__":
    main()
