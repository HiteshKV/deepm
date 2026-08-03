"""Reconcile simulator or broker state."""

from __future__ import annotations

import argparse

from deepm.live.brokers import IBKRBroker
from deepm.live.config import (
    DEFAULT_CONFIG,
    ledger_path_for_mode,
    load_live_config,
    path_from_config,
    starting_cash_for_mode,
)
from deepm.live.instruments import load_instruments
from deepm.live.ledger import PortfolioStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile DeePM live positions")
    parser.add_argument("--mode", choices=["sim", "ibkr-paper", "ibkr-live"], default="sim")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_live_config(args.config)
    instruments = load_instruments(path_from_config(config, "paths", "instrument_map"))
    if args.mode == "sim" or (
        args.mode == "ibkr-paper" and bool(config.get("ibkr", {}).get("paper_simulation", True))
    ):
        store = PortfolioStore(ledger_path_for_mode(config, args.mode), starting_cash_for_mode(config, args.mode))
        print(f"cash={store.cash():.2f}")
        print("positions require latest prices; run the daily report for mark-to-market values")
        return

    broker = IBKRBroker(args.mode, config.get("ibkr", {}), instruments)
    broker.current_positions({})


if __name__ == "__main__":
    main()
