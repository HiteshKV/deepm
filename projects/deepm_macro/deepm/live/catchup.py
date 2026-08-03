"""Idempotent sequential catch-up for the DeePM paper ledger."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Mapping

from deepm.live.backfill import run_backfill
from deepm.live.config import (
    DEFAULT_CONFIG,
    ledger_path_for_mode,
    load_live_config,
    starting_cash_for_mode,
)
from deepm.live.ledger import PortfolioStore


def _scheduled_utc(config: Mapping[str, object]) -> time:
    value = str(config.get("schedule", {}).get("run_after_utc", "22:15"))
    hour, minute = value.split(":", 1)
    return time(hour=int(hour), minute=int(minute), tzinfo=timezone.utc)


def previous_business_date(value: date) -> date:
    """Return the nearest weekday on or before ``value``."""
    candidate = value
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def latest_eligible_date(
    config: Mapping[str, object],
    now: datetime | None = None,
) -> date:
    """Return the latest weekday whose configured EOD gate has opened."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    candidate = current.date()
    if candidate.weekday() >= 5:
        return previous_business_date(candidate)
    if current.timetz() < _scheduled_utc(config):
        candidate -= timedelta(days=1)
    return previous_business_date(candidate)


def next_business_date(value: date) -> date:
    """Return the first weekday strictly after ``value``."""
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def catch_up(
    mode: str,
    config_path: str | Path = DEFAULT_CONFIG,
    end_date: date | None = None,
    now: datetime | None = None,
    send_email: bool = False,
    no_place_orders: bool = False,
) -> Path | None:
    """Run every unprocessed weekday through the latest eligible EOD date."""
    if mode not in {"sim", "ibkr-paper"}:
        raise ValueError("Sequential catch-up is supported only for local sim and ibkr-paper modes")

    config = load_live_config(config_path)
    if mode == "ibkr-paper" and not bool(config.get("ibkr", {}).get("paper_simulation", True)):
        raise ValueError("Catch-up refuses broker-routed ibkr-paper mode")

    target = end_date or latest_eligible_date(config, now)
    store = PortfolioStore(
        ledger_path_for_mode(config, mode),
        starting_cash_for_mode(config, mode),
    )
    last_completed = store.latest_completed_run(mode)
    if last_completed is not None and last_completed >= target:
        return None

    start = target if last_completed is None else next_business_date(last_completed)
    if start > target:
        return None
    return run_backfill(
        mode=mode,
        config_path=config_path,
        start_date=start,
        end_date=target,
        reset_ledger=False,
        no_place_orders=no_place_orders,
        send_email=send_email,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume the DeePM paper ledger through the latest completed EOD date"
    )
    parser.add_argument("--mode", choices=["sim", "ibkr-paper"], default="ibkr-paper")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--no-place-orders", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = catch_up(
        mode=args.mode,
        config_path=args.config,
        end_date=date.fromisoformat(args.end_date) if args.end_date else None,
        send_email=args.send_email,
        no_place_orders=args.no_place_orders,
    )
    print(output if output is not None else "up_to_date")


if __name__ == "__main__":
    main()
