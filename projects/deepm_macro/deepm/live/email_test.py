"""Send a DeePM report email to validate provider credentials."""

from __future__ import annotations

import argparse
from datetime import date

from deepm.live.config import DEFAULT_CONFIG, load_live_config, path_from_config
from deepm.live.exceptions import LiveTradingError
from deepm.live.reporting import send_email_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a DeePM report email as a provider test")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", default="ibkr-paper")
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_date = date.fromisoformat(args.date)
    config = load_live_config(args.config)
    report_path = path_from_config(config, "paths", "run_dir") / run_date.isoformat() / "report.html"
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")
    try:
        send_email_report(report_path, config.get("email", {}), run_date, args.mode)
    except LiveTradingError as exc:
        raise SystemExit(f"Email send failed: {exc}") from exc
    print(f"sent {report_path}")


if __name__ == "__main__":
    main()
