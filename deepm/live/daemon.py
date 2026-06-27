"""Autonomous daily runner for DeePM live/paper management."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Mapping

from deepm.live.config import DEFAULT_CONFIG, load_live_config, path_from_config
from deepm.live.daily import run_daily


def parse_utc_time(value: str) -> dt_time:
    """Parse a HH:MM UTC schedule value."""
    hour, minute = value.split(":", 1)
    return dt_time(hour=int(hour), minute=int(minute), tzinfo=timezone.utc)


def should_run_now(config: Mapping[str, object], now: datetime) -> bool:
    """Return true when the configured daily schedule has opened."""
    schedule = config.get("schedule", {})
    if bool(schedule.get("weekdays_only", True)) and now.weekday() >= 5:
        return False
    scheduled = parse_utc_time(str(schedule.get("run_after_utc", "22:15")))
    return now.timetz() >= scheduled


def status_path(config: Mapping[str, object], run_date: date) -> Path:
    """Return the status path for a daily run."""
    return path_from_config(config, "paths", "run_dir") / run_date.isoformat() / "status.json"


def completed_for_date(config: Mapping[str, object], run_date: date) -> bool:
    """Check whether any previous daemon run completed for this date."""
    path = status_path(config, run_date)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") in {"completed", "already_ran", "dry_run"}


def run_once(args: argparse.Namespace) -> Path | None:
    """Run immediately or when the schedule is open."""
    config = load_live_config(args.config)
    now = datetime.now(timezone.utc)
    run_date = date.fromisoformat(args.date) if args.date else now.date()
    if not args.run_now and not should_run_now(config, now):
        return None
    if not args.force and completed_for_date(config, run_date):
        return status_path(config, run_date).with_name("report.html")
    return run_daily(
        mode=args.mode,
        config_path=args.config,
        run_date=run_date,
        send_email=args.send_email,
        no_place_orders=args.no_place_orders,
        force=args.force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeePM live management on a daily schedule")
    parser.add_argument("--mode", choices=["sim", "ibkr-paper", "ibkr-live"], default="ibkr-paper")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--date", default=None)
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--no-place-orders", action="store_true")
    parser.add_argument("--run-now", action="store_true", help="Ignore the configured time gate once")
    parser.add_argument("--force", action="store_true", help="Run even if today's report already exists")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    parser.add_argument("--poll-seconds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_live_config(args.config)
    poll_seconds = int(args.poll_seconds or config.get("schedule", {}).get("poll_seconds", 300))

    if args.once:
        report = run_once(args)
        if report:
            print(report)
        else:
            print("schedule_not_open")
        return

    print(f"DeePM live daemon started mode={args.mode} poll_seconds={poll_seconds}")
    while True:
        try:
            report = run_once(args)
            if report:
                print(f"{datetime.now(timezone.utc).isoformat()} report={report}", flush=True)
        except Exception as exc:  # keep daemon alive but visible
            print(f"{datetime.now(timezone.utc).isoformat()} error={exc}", flush=True)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
