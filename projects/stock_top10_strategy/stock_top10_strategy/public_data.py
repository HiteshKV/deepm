"""Public proxy data builder for historical top-10 market-cap rankings."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from stock_top10_strategy.config import PROJECT_ROOT


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_public_corporations_by_market_capitalization"
USER_AGENT = "deepm-top10-strategy/1.0 (local research data builder)"
QUARTER_ENDS = {
    "First quarter": "03-31",
    "Second quarter": "06-30",
    "Third quarter": "09-30",
    "Fourth quarter": "12-31",
}
FX_SYMBOLS = {
    "USD": None,
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "JPY": ("JPY=X", True),
    "HKD": ("HKD=X", True),
    "KRW": ("KRW=X", True),
    "CHF": ("CHF=X", True),
    "RUB": ("RUB=X", True),
    "BRL": ("BRL=X", True),
}


def build_wikipedia_yahoo_dataset(
    output_path: str | Path,
    audit_output_path: str | Path,
    mapping_path: str | Path,
    start_year: int = 2000,
    end_year: int | None = None,
    top_n: int = 10,
) -> tuple[Path, Path]:
    """Build a public proxy top-10 dataset from Wikipedia rankings and Yahoo prices."""
    output_path = _project_path(output_path)
    audit_output_path = _project_path(audit_output_path)
    mapping_path = _project_path(mapping_path)
    end_year = end_year or date.today().year

    rankings = fetch_wikipedia_rankings(start_year=start_year, end_year=end_year, top_n=top_n)
    mapping = pd.read_csv(mapping_path).fillna("")
    mapped = rankings.merge(mapping, on="company_name", how="left", suffixes=("", "_map"))
    mapped["security_id"] = mapped["security_id"].replace("", np.nan)
    mapped["yahoo_symbol"] = mapped["yahoo_symbol"].replace("", np.nan)
    mapped["currency"] = mapped["currency"].replace("", np.nan).fillna("USD")
    mapped["country"] = mapped["country"].replace("", np.nan).fillna("unknown")

    start = f"{start_year}-01-01"
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=7)).date().isoformat()
    prices = _download_price_tables(mapped["yahoo_symbol"].dropna().unique(), start, end)
    fx = _download_fx_tables(mapped["currency"].dropna().unique(), start, end)

    valid_mapped = mapped.dropna(subset=["security_id", "yahoo_symbol"]).copy()
    valid_mapped["security_id"] = valid_mapped["security_id"].astype(str)
    security_meta = (
        valid_mapped.sort_values("ranking_date")
        .drop_duplicates("security_id", keep="last")
        .set_index("security_id", drop=False)
    )

    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    snapshots = sorted(mapped["ranking_date"].drop_duplicates())
    business_days = pd.bdate_range(min(snapshots), pd.Timestamp.today().normalize())

    for day in business_days:
        snapshot_date = max(d for d in snapshots if d <= day)
        current_snapshot = valid_mapped[valid_mapped["ranking_date"] == snapshot_date].sort_values("rank")
        active_by_id = {
            str(item.security_id): item
            for item in current_snapshot.itertuples(index=False)
        }
        known = valid_mapped[valid_mapped["ranking_date"] <= snapshot_date].sort_values("ranking_date")
        known_by_id = known.drop_duplicates("security_id", keep="last").set_index("security_id", drop=False)

        for security_id, meta_row in known_by_id.iterrows():
            active_item = active_by_id.get(str(security_id))
            item = active_item if active_item is not None else security_meta.loc[security_id]
            symbol = _item_value(item, "yahoo_symbol")
            if pd.isna(symbol) or symbol not in prices:
                missing.append(_missing_record(day, item, "missing_mapping_or_price_table"))
                continue
            price_table = prices[symbol]
            if day not in price_table.index:
                missing.append(_missing_record(day, item, "missing_price_for_date"))
                continue
            price = price_table.loc[day]
            currency = _item_value(item, "currency") or "USD"
            fx_to_usd = _fx_value(fx, currency, day)
            if fx_to_usd is None:
                missing.append(_missing_record(day, item, "missing_fx_for_date"))
                continue
            is_active = active_item is not None
            rank = int(getattr(active_item, "rank")) if active_item is not None else ""
            market_cap = (
                float(_item_value(active_item, "market_cap_usd"))
                if active_item is not None
                else float(meta_row["market_cap_usd"])
            )
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "security_id": security_id,
                    "ticker": symbol,
                    "company_name": _item_value(item, "company_name"),
                    "country": _item_value(item, "country"),
                    "currency": currency,
                    "market_cap_usd": market_cap,
                    "open": float(price["Open"]),
                    "close": float(price["Close"]),
                    "adj_close": float(price["Adj Close"]),
                    "fx_to_usd": float(fx_to_usd),
                    "split_factor": 1.0,
                    "dividend": 0.0,
                    "is_active": is_active,
                    "delist_date": "",
                    "ranking_date": snapshot_date.date().isoformat(),
                    "rank": rank,
                    "source": "wikipedia_rankings_yahoo_prices",
                }
            )

    data = pd.DataFrame(rows).drop_duplicates(["date", "security_id"]).sort_values(["date", "rank", "security_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    active_data = data[data["is_active"]] if not data.empty else data
    coverage = active_data.groupby("date")["security_id"].nunique().describe().to_dict() if not active_data.empty else {}
    missing_frame = pd.DataFrame(missing)
    audit = {
        "source_url": WIKIPEDIA_URL,
        "start_year": start_year,
        "end_year": end_year,
        "top_n_requested": top_n,
        "ranking_snapshots": len(snapshots),
        "output_rows": int(len(data)),
        "output_dates": int(data["date"].nunique()) if not data.empty else 0,
        "output_securities": int(data["security_id"].nunique()) if not data.empty else 0,
        "active_members_per_date_summary": coverage,
        "unmapped_companies": sorted(mapped.loc[mapped["yahoo_symbol"].isna(), "company_name"].dropna().unique().tolist()),
        "missing_records_by_reason": (
            missing_frame["reason"].value_counts().to_dict() if not missing_frame.empty else {}
        ),
        "caveat": (
            "This is a public proxy dataset. Membership comes from Wikipedia ranking snapshots, "
            "not a daily point-in-time vendor security master. Yahoo prices and FX are best-effort."
        ),
    }
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return output_path, audit_output_path


def fetch_wikipedia_rankings(start_year: int, end_year: int, top_n: int = 10) -> pd.DataFrame:
    """Fetch and parse public top market-cap ranking snapshots from Wikipedia."""
    response = requests.get(WIKIPEDIA_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows: list[dict[str, object]] = []
    for heading in soup.find_all(["h2", "h3"]):
        year_text = heading.get_text(" ", strip=True)
        if not re.fullmatch(r"20\d{2}", year_text):
            continue
        year = int(year_text)
        if year < start_year or year > end_year:
            continue
        table = heading.find_next("table")
        if table is None:
            continue
        header = [_clean_text(cell.get_text(" ", strip=True)) for cell in table.find_all("tr")[0].find_all(["th", "td"])]
        if "Name" in header:
            rows.extend(_parse_annual_table(table, year, top_n))
        else:
            rows.extend(_parse_quarterly_table(table, year, header, top_n))
    if not rows:
        raise RuntimeError("No ranking rows parsed from Wikipedia")
    return pd.DataFrame(rows).sort_values(["ranking_date", "rank"]).reset_index(drop=True)


def _parse_annual_table(table, year: int, top_n: int) -> list[dict[str, object]]:
    rows = []
    ranking_date = pd.Timestamp(f"{year}-03-31")
    for tr in table.find_all("tr")[1:]:
        cells = [_clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if len(cells) < 5:
            continue
        rank = _parse_int(cells[0])
        if rank is None or rank > top_n:
            continue
        rows.append(
            {
                "ranking_date": ranking_date,
                "year": year,
                "quarter": "Annual",
                "rank": rank,
                "company_name": _normalize_company(cells[1]),
                "market_cap_usd": _parse_market_cap_millions(cells[4]) * 1_000_000,
            }
        )
    return rows


def _parse_quarterly_table(table, year: int, header: list[str], top_n: int) -> list[dict[str, object]]:
    quarter_headers = [_quarter_name(h) for h in header]
    quarter_headers = [h for h in quarter_headers if h is not None]
    rows = []
    for tr in table.find_all("tr")[1:]:
        raw_cells = [c.get_text("\n", strip=True).replace("\xa0", " ") for c in tr.find_all(["th", "td"])]
        cells = [cell for cell in raw_cells if _clean_text(cell)]
        if not cells:
            continue
        rank = _parse_int(cells[0])
        if rank is None or rank > top_n:
            continue
        for quarter, cell in zip(quarter_headers, cells[1:]):
            parsed = _parse_company_value_cell(cell)
            if parsed is None:
                continue
            company, value = parsed
            rows.append(
                {
                    "ranking_date": pd.Timestamp(f"{year}-{QUARTER_ENDS[quarter]}"),
                    "year": year,
                    "quarter": quarter,
                    "rank": rank,
                    "company_name": _normalize_company(company),
                    "market_cap_usd": value * 1_000_000,
                }
            )
    return rows


def _parse_company_value_cell(cell: str) -> tuple[str, float] | None:
    lines = [
        _clean_text(line)
        for line in cell.split("\n")
        if _clean_text(line) and not re.fullmatch(r"\[|\]", _clean_text(line))
    ]
    numeric_index = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"[\d,]+(?:\.\d+)?", line):
            numeric_index = i
            break
    if numeric_index is None or numeric_index == 0:
        return None
    company = " ".join(lines[:numeric_index])
    return company, _parse_market_cap_millions(lines[numeric_index])


def _download_price_tables(symbols, start: str, end: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for symbol in sorted({str(s) for s in symbols if str(s) != "nan"}):
        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        needed = ["Open", "Close", "Adj Close"]
        if not set(needed).issubset(data.columns):
            continue
        data = data[needed].dropna()
        data.index = pd.to_datetime(data.index).normalize()
        data = data[~data.index.duplicated(keep="last")]
        data = data.reindex(pd.bdate_range(start, end)).ffill().dropna()
        tables[symbol] = data
    return tables


def _download_fx_tables(currencies, start: str, end: str) -> dict[str, pd.Series]:
    tables: dict[str, pd.Series] = {"USD": pd.Series(1.0, index=pd.bdate_range(start, end))}
    for currency in sorted(set(currencies)):
        if currency == "USD" or currency not in FX_SYMBOLS:
            continue
        symbol_info = FX_SYMBOLS[currency]
        if symbol_info is None:
            continue
        symbol, invert = symbol_info
        data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if data.empty:
            continue
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).normalize()
        series = (1.0 / close) if invert else close
        tables[currency] = series.reindex(pd.bdate_range(start, end)).ffill()
    return tables


def _fx_value(fx: dict[str, pd.Series], currency: str, day: pd.Timestamp) -> float | None:
    if currency == "USD":
        return 1.0
    series = fx.get(currency)
    if series is None or day not in series.index or pd.isna(series.loc[day]):
        return None
    return float(series.loc[day])


def _missing_record(day: pd.Timestamp, item, reason: str) -> dict[str, object]:
    return {
        "date": day.date().isoformat(),
        "ranking_date": _item_value(item, "ranking_date").date().isoformat(),
        "rank": int(_item_value(item, "rank")),
        "company_name": _item_value(item, "company_name"),
        "yahoo_symbol": _item_value(item, "yahoo_symbol", None),
        "reason": reason,
    }


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _item_value(item, key: str, default=None):
    if isinstance(item, pd.Series):
        return item.get(key, default)
    return getattr(item, key, default)


def _clean_text(value: str) -> str:
    value = re.sub(r"\[\s*\d+\s*\]", "", value)
    value = re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()
    return value


def _normalize_company(value: str) -> str:
    value = _clean_text(value)
    replacements = {
        "Petro China": "PetroChina",
        "Petrochina": "PetroChina",
        "Intel Corporation": "Intel",
    }
    return replacements.get(value, value)


def _parse_market_cap_millions(value: str) -> float:
    cleaned = re.sub(r"[^\d.]", "", value)
    if not cleaned:
        raise ValueError(f"Could not parse market cap: {value}")
    return float(cleaned)


def _parse_int(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _quarter_name(value: str) -> str | None:
    for quarter in QUARTER_ENDS:
        if value.startswith(quarter):
            return quarter
    return None
