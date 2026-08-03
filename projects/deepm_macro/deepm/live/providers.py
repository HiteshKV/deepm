"""Market data providers for simulation and broker-backed modes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from deepm.live.exceptions import BrokerUnavailable, DataUnavailable
from deepm.live.types import Instrument


class DataProvider:
    """Base market data provider interface."""

    def load_history(
        self,
        instruments: dict[str, Instrument],
        start_date: str,
        end_date: date,
    ) -> pd.DataFrame:
        raise NotImplementedError


class LocalParquetProvider(DataProvider):
    """Read the local raw price snapshot used by the simulator."""

    def __init__(self, parquet_path: str | Path):
        self.parquet_path = Path(parquet_path)

    def load_history(
        self,
        instruments: dict[str, Instrument],
        start_date: str,
        end_date: date,
    ) -> pd.DataFrame:
        if not self.parquet_path.exists():
            raise DataUnavailable(f"Raw price parquet not found: {self.parquet_path}")
        data = pd.read_parquet(self.parquet_path)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        tickers = [ticker for ticker in instruments if ticker in data.columns]
        if not tickers:
            raise DataUnavailable("No configured live tickers were found in the raw price parquet")
        data = data.loc[pd.Timestamp(start_date): pd.Timestamp(end_date), tickers]
        if data.empty:
            raise DataUnavailable(
                f"No raw price rows available through {end_date.isoformat()} in {self.parquet_path}"
            )
        return data


class YahooEODProvider(DataProvider):
    """Download EOD history from Yahoo for simulation data refreshes."""

    def load_history(
        self,
        instruments: dict[str, Instrument],
        start_date: str,
        end_date: date,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataUnavailable("yfinance is required for yahoo_eod provider") from exc

        yahoo_to_model = {
            instrument.yahoo_symbol: ticker
            for ticker, instrument in instruments.items()
            if instrument.yahoo_symbol
        }
        raw = yf.download(
            list(yahoo_to_model),
            start=start_date,
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.rename(columns=yahoo_to_model)
        close.index = pd.to_datetime(close.index)
        return close[[ticker for ticker in instruments if ticker in close.columns]]


class IBKRMarketDataProvider(DataProvider):
    """Placeholder provider that fails closed until an IBKR session is wired in."""

    def load_history(
        self,
        instruments: dict[str, Instrument],
        start_date: str,
        end_date: date,
    ) -> pd.DataFrame:
        raise BrokerUnavailable(
            "IBKR market data provider requires a connected TWS/Gateway session. "
            "Run simulator mode first, then wire IBKR credentials and market data subscriptions."
        )


def provider_from_name(name: str, raw_price_parquet: str | Path) -> DataProvider:
    """Factory for configured providers."""
    if name == "local_parquet":
        return LocalParquetProvider(raw_price_parquet)
    if name == "yahoo_eod":
        return YahooEODProvider()
    if name == "ibkr_market_data":
        return IBKRMarketDataProvider()
    raise DataUnavailable(f"Unknown data provider: {name}")
