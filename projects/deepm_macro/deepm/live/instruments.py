"""Instrument map loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from deepm.live.types import Instrument


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_instruments(path: str | Path) -> dict[str, Instrument]:
    """Load the configured model ticker to broker instrument map."""
    frame = pd.read_csv(path)
    instruments: dict[str, Instrument] = {}
    for row in frame.to_dict(orient="records"):
        instrument = Instrument(
            ticker=str(row["ticker"]),
            bloomberg_ticker=str(row["bloomberg_ticker"]),
            yahoo_symbol=str(row["yahoo_symbol"]),
            description=str(row["description"]),
            ibkr_symbol=str(row["ibkr_symbol"]),
            sec_type=str(row["sec_type"]),
            exchange=str(row["exchange"]),
            currency=str(row["currency"]),
            point_value=float(row["point_value"]),
            tick_size=float(row["tick_size"]),
            contract_type=str(row["contract_type"]),
            max_contracts=int(row["max_contracts"]),
            reviewed=_parse_bool(row["reviewed"]),
        )
        instruments[instrument.ticker] = instrument
    return instruments


def tickers_from_instruments(instruments: Iterable[Instrument]) -> list[str]:
    """Return model tickers in deterministic instrument-map order."""
    return [instrument.ticker for instrument in instruments]
