"""Replay the daily live pipeline over a historical date range."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from deepm.live.config import (
    DEFAULT_CONFIG,
    ledger_path_for_mode,
    load_live_config,
    mode_provider,
    path_from_config,
    resolve_project_path,
)
from deepm.live.daily import run_daily
from deepm.live.data_update import latest_snapshot_path, persist_live_snapshot
from deepm.live.instruments import load_instruments
from deepm.live.providers import provider_from_name


def business_dates(start_date: date, end_date: date) -> list[date]:
    """Return business dates for replay."""
    return [ts.date() for ts in pd.bdate_range(start_date, end_date)]


def _default_start(end_date: date, lookback_days: int) -> date:
    return (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).date()


def _reset_ledger(config: dict[str, object], mode: str) -> Path | None:
    ledger = ledger_path_for_mode(config, mode)
    if not ledger.exists():
        return None
    backup = ledger.with_suffix(
        f".{datetime.now().strftime('%Y%m%d%H%M%S')}.bak{ledger.suffix}"
    )
    shutil.move(str(ledger), str(backup))
    return backup


def _write_backfill_config(
    config: dict[str, object],
    mode: str,
    snapshot_path: Path,
) -> Path:
    cfg = dict(config)
    cfg.pop("_config_path", None)
    cfg["data"] = dict(cfg.get("data", {}))
    if mode == "sim":
        cfg["data"]["sim_provider"] = "local_parquet"
    elif mode == "ibkr-paper":
        cfg["data"]["paper_provider"] = "local_parquet"
    cfg["data"]["raw_price_parquet"] = str(snapshot_path)
    cfg["data"]["persist_live_snapshot"] = False

    state_dir = path_from_config(config, "paths", "state_dir")
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"backfill_{mode.replace('-', '_')}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run_backfill(
    mode: str,
    config_path: str | Path,
    start_date: date,
    end_date: date,
    reset_ledger: bool = False,
    no_place_orders: bool = False,
) -> Path:
    """Replay daily runs and write a compact summary CSV."""
    config = load_live_config(config_path)
    instruments = load_instruments(path_from_config(config, "paths", "instrument_map"))
    raw_price_path = resolve_project_path(config["data"]["raw_price_parquet"])
    provider_name = mode_provider(config, mode)
    provider = provider_from_name(provider_name, raw_price_path)

    raw_prices = provider.load_history(
        instruments,
        start_date=str(config["data"]["history_start_date"]),
        end_date=end_date,
    )
    snapshot_path = persist_live_snapshot(raw_prices, config, end_date, provider_name)
    replay_config = _write_backfill_config(config, mode, latest_snapshot_path(config))

    backup = _reset_ledger(config, mode) if reset_ledger else None
    rows: list[dict[str, object]] = []
    for run_date in business_dates(start_date, end_date):
        report = run_daily(
            mode=mode,
            config_path=replay_config,
            run_date=run_date,
            send_email=False,
            no_place_orders=no_place_orders,
        )
        run_dir = report.parent
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        summary_path = run_dir / "summary.json"
        summary_row = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        orders = pd.read_csv(run_dir / "orders.csv")
        fills = pd.read_csv(run_dir / "fills.csv")
        positions = pd.read_csv(run_dir / "positions.csv")
        rows.append(
            {
                "date": run_date.isoformat(),
                "status": status.get("status"),
                "errors": " | ".join(status.get("errors", [])),
                "orders_count": len(orders),
                "fills_count": len(fills),
                "positions_count": len(positions),
                **summary_row,
            }
        )

    output_dir = path_from_config(config, "paths", "run_dir") / "backfills"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"backfill_{mode}_{start_date}_{end_date}.csv"
    summary = pd.DataFrame(rows)
    if backup is not None:
        summary["ledger_backup"] = str(backup)
    summary["snapshot_path"] = str(snapshot_path)
    summary.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay DeePM paper/sim live runs over a date range")
    parser.add_argument("--mode", choices=["sim", "ibkr-paper"], default="ibkr-paper")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=31)
    parser.add_argument("--reset-ledger", action="store_true")
    parser.add_argument("--no-place-orders", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else _default_start(end_date, args.lookback_days)
    )
    output = run_backfill(
        mode=args.mode,
        config_path=args.config,
        start_date=start_date,
        end_date=end_date,
        reset_ledger=args.reset_ledger,
        no_place_orders=args.no_place_orders,
    )
    print(output)


if __name__ == "__main__":
    main()
