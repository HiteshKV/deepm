"""Yahoo price-fill helper for point-in-time membership files.

This helper does not reconstruct historical top-10 membership. It only fills
price columns for a user-supplied point-in-time security/date/membership file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_REQUIRED_COLUMNS = {
    "date",
    "security_id",
    "ticker",
    "company_name",
    "country",
    "currency",
    "market_cap_usd",
}


def fill_yahoo_prices(input_path: str | Path, output_path: str | Path) -> Path:
    """Fill open/close/adj_close fields using Yahoo Finance for supplied tickers."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame = pd.read_csv(input_path)
    missing = sorted(BASE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing membership columns for Yahoo price fill: {missing}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    tickers = sorted(frame["ticker"].dropna().astype(str).unique())
    start = frame["date"].min().date().isoformat()
    end = (frame["date"].max() + pd.Timedelta(days=7)).date().isoformat()
    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    price_rows = []
    for ticker in tickers:
        table = _ticker_table(prices, ticker, len(tickers) == 1)
        table = table.reset_index()
        table["date"] = pd.to_datetime(table["Date"]).dt.normalize()
        table["ticker"] = ticker
        price_rows.append(
            table[["date", "ticker", "Open", "Close", "Adj Close"]].rename(
                columns={"Open": "open", "Close": "close", "Adj Close": "adj_close"}
            )
        )
    price_frame = pd.concat(price_rows, ignore_index=True)
    merged = frame.drop(columns=[c for c in ["open", "close", "adj_close"] if c in frame.columns]).merge(
        price_frame,
        on=["date", "ticker"],
        how="left",
    )
    if merged[["open", "close", "adj_close"]].isna().any().any():
        missing_rows = merged.loc[merged[["open", "close", "adj_close"]].isna().any(axis=1), ["date", "ticker"]]
        raise ValueError(f"Yahoo did not return all requested prices: {missing_rows.head().to_dict('records')}")
    merged["fx_to_usd"] = merged.get("fx_to_usd", 1.0)
    merged["split_factor"] = merged.get("split_factor", 1.0)
    merged["dividend"] = merged.get("dividend", 0.0)
    merged["is_active"] = merged.get("is_active", True)
    merged["delist_date"] = merged.get("delist_date", "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def _ticker_table(prices: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
    if single_ticker:
        return prices
    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Unexpected Yahoo response shape for multiple tickers")
    if ticker not in prices.columns.get_level_values(0):
        raise ValueError(f"Yahoo response missing ticker: {ticker}")
    return prices[ticker]
