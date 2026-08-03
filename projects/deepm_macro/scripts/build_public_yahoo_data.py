#!/usr/bin/env python
"""Build a public-data proxy for the README raw price panel.

The paper's vendor-sourced continuous futures panel is not redistributed. This
script downloads Yahoo Finance-accessible futures, FX, index, and ETF proxies
and writes the wide parquet expected by ``scripts/prepare_features.py``:

    data/data_20260625.parquet

The output is useful for local pipeline runs and smoke tests, but it is not a
licensed Bloomberg/Refinitiv-equivalent panel and should not be treated as a
paper-reproduction dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from deepm._paths import DATA_DIR


DEFAULT_START_DATE = "1990-01-02"
DEFAULT_END_DATE = "2026-06-25"
DEFAULT_SNAPSHOT = DEFAULT_END_DATE.replace("-", "")


@dataclass(frozen=True)
class SourceSpec:
    ticker: str
    asset: str
    yahoo_symbol: str
    source_kind: str
    note: str = ""


SOURCES = [
    SourceSpec("EN", "Nasdaq 100", "NQ=F", "futures"),
    SourceSpec("ER", "Russell 2000", "^RUT", "index", "Proxy for RTY futures; longer public history."),
    SourceSpec("ES", "S&P 500", "ES=F", "futures"),
    SourceSpec("YM", "Dow Jones", "YM=F", "futures"),
    SourceSpec("LX", "FTSE 100", "^FTSE", "index", "Proxy for FTSE futures."),
    SourceSpec("CA", "CAC 40", "^FCHI", "index", "Proxy for CAC futures."),
    SourceSpec("XU", "EuroStoxx 50", "^STOXX50E", "index", "Proxy for EuroStoxx futures."),
    SourceSpec("NK", "Nikkei 225", "^N225", "index", "Proxy for Nikkei futures."),
    SourceSpec("HS", "Hang Seng", "^HSI", "index", "Proxy for Hang Seng futures."),
    SourceSpec("FB", "US 5-Year Note", "ZF=F", "futures"),
    SourceSpec("TU", "US 2-Year Note", "ZT=F", "futures"),
    SourceSpec("TY", "US 10-Year Note", "ZN=F", "futures"),
    SourceSpec("US", "US 30-Year Bond", "ZB=F", "futures"),
    SourceSpec("DT", "German Bund 10yr", "EXHD.DE", "etf", "German government 5.5-10.5y ETF proxy."),
    SourceSpec("UB", "German Bobl 5yr", "EXHC.DE", "etf", "German government 2.5-5.5y ETF proxy."),
    SourceSpec("UZ", "German Schatz 2yr", "EXHB.DE", "etf", "German government 1.5-2.5y ETF proxy."),
    SourceSpec("GS", "UK Gilt", "IGLT.L", "etf", "UK gilt ETF proxy."),
    SourceSpec("CB", "Canada 10-Year", "XGB.TO", "etf", "Canadian government bond ETF proxy."),
    SourceSpec("BC", "Brent Crude", "BZ=F", "futures"),
    SourceSpec("BG", "Gasoil", "HO=F", "futures", "NY Harbor ULSD/heating oil proxy for gasoil."),
    SourceSpec("ZB", "RBOB Gasoline", "RB=F", "futures"),
    SourceSpec("ZN", "Natural Gas", "NG=F", "futures"),
    SourceSpec("ZU", "WTI Crude", "CL=F", "futures"),
    SourceSpec("ZA", "Palladium", "PA=F", "futures"),
    SourceSpec("ZG", "Gold", "GC=F", "futures"),
    SourceSpec("ZI", "Silver", "SI=F", "futures"),
    SourceSpec("ZP", "Platinum", "PL=F", "futures"),
    SourceSpec("ZK", "Copper", "HG=F", "futures"),
    SourceSpec("KW", "KC Wheat", "KE=F", "futures"),
    SourceSpec("ZC", "Corn", "ZC=F", "futures"),
    SourceSpec("ZL", "Soybean Oil", "ZL=F", "futures"),
    SourceSpec("ZM", "Soybean Meal", "ZM=F", "futures"),
    SourceSpec("ZS", "Soybeans", "ZS=F", "futures"),
    SourceSpec("ZW", "Chicago Wheat", "ZW=F", "futures"),
    SourceSpec("CC", "Cocoa", "CC=F", "futures"),
    SourceSpec("CT", "Cotton", "CT=F", "futures"),
    SourceSpec("JO", "Orange Juice", "OJ=F", "futures"),
    SourceSpec("KC", "Coffee", "KC=F", "futures"),
    SourceSpec("SB", "Sugar", "SB=F", "futures"),
    SourceSpec("ZF", "Feeder Cattle", "GF=F", "futures"),
    SourceSpec("ZT", "Live Cattle", "LE=F", "futures"),
    SourceSpec("ZZ", "Lean Hogs", "HE=F", "futures"),
    SourceSpec("AN", "AUD/USD", "6A=F", "futures"),
    SourceSpec("BN", "GBP/USD", "6B=F", "futures"),
    SourceSpec("CN", "CAD/USD", "6C=F", "futures"),
    SourceSpec("FN", "EUR/USD", "6E=F", "futures"),
    SourceSpec("JN", "JPY/USD", "6J=F", "futures"),
    SourceSpec("SN", "CHF/USD", "6S=F", "futures"),
    SourceSpec("MP", "MXN/USD", "6M=F", "futures"),
    SourceSpec("DX", "US Dollar Index", "DX-Y.NYB", "index"),
]


def _extract_close(history: pd.DataFrame) -> pd.Series:
    """Return an adjusted close when available, falling back to close."""
    if history.empty:
        return pd.Series(dtype="float64")

    if isinstance(history.columns, pd.MultiIndex):
        price_level = history.columns.names.index("Price") if "Price" in history.columns.names else 0
        column = "Adj Close" if "Adj Close" in history.columns.get_level_values(price_level) else "Close"
        selected = history.xs(column, axis=1, level=price_level)
        close = selected.iloc[:, 0].copy() if isinstance(selected, pd.DataFrame) else selected.copy()
    else:
        column = "Adj Close" if "Adj Close" in history.columns else "Close"
        close = history[column].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.astype("float64")


def _download_one(spec: SourceSpec, start_date: str, end_date: str) -> pd.Series:
    history = yf.download(
        spec.yahoo_symbol,
        start=start_date,
        end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    close = _extract_close(history)
    close = close[close > 0.0]
    close.name = spec.ticker
    return close


def build_panel(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = pd.bdate_range(start_date, end_date)
    panel = pd.DataFrame(index=calendar)
    audit_rows = []

    for spec in SOURCES:
        print(f"{spec.ticker:>2} <- {spec.yahoo_symbol:<10} {spec.asset}")
        close = _download_one(spec, start_date, end_date)
        panel[spec.ticker] = close.reindex(calendar)

        observed = panel[spec.ticker].dropna()
        audit_rows.append(
            {
                "ticker": spec.ticker,
                "asset": spec.asset,
                "yahoo_symbol": spec.yahoo_symbol,
                "source_kind": spec.source_kind,
                "first_date": observed.index.min().date().isoformat() if len(observed) else "",
                "last_date": observed.index.max().date().isoformat() if len(observed) else "",
                "rows": int(len(observed)),
                "coverage_pct": round(float(len(observed) / len(calendar) * 100.0), 2),
                "note": spec.note,
            }
        )

    return panel, pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public Yahoo proxy data for DeePM")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Inclusive start date for the business-day panel.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Inclusive end date for the business-day panel.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / f"data_{DEFAULT_SNAPSHOT}.parquet"),
        help="Wide raw price parquet output path.",
    )
    parser.add_argument(
        "--sources-output",
        default=str(DATA_DIR / f"data_{DEFAULT_SNAPSHOT}_sources.csv"),
        help="CSV audit of source symbols and coverage.",
    )
    parser.add_argument(
        "--json-output",
        default=str(DATA_DIR / f"data_{DEFAULT_SNAPSHOT}_sources.json"),
        help="JSON audit of source symbols and coverage.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel, audit = build_panel(args.start_date, args.end_date)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)

    audit.to_csv(args.sources_output, index=False)
    Path(args.json_output).write_text(
        json.dumps(audit.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )

    print("")
    print(f"Saved raw panel: {output} shape={panel.shape}")
    print(f"Saved source audit: {args.sources_output}")
    print(audit[["ticker", "yahoo_symbol", "first_date", "last_date", "coverage_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
