"""Point-in-time data loading and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_top10_strategy.config import StrategyConfig


REQUIRED_COLUMNS = {
    "date",
    "security_id",
    "ticker",
    "company_name",
    "country",
    "currency",
    "market_cap_usd",
    "close",
    "adj_close",
    "fx_to_usd",
    "split_factor",
    "dividend",
    "is_active",
    "delist_date",
}

OPTIONAL_COLUMNS = {"open"}


class DataValidationError(ValueError):
    """Raised when point-in-time strategy data is invalid."""


def load_point_in_time_data(config: StrategyConfig) -> pd.DataFrame:
    """Load CSV or Parquet point-in-time stock data."""
    path = config.data_path
    if not path.exists():
        raise DataValidationError(
            f"Data file not found: {path}. Provide a point-in-time CSV/Parquet export."
        )
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise DataValidationError(f"Unsupported data file type: {path.suffix}")

    return validate_point_in_time_data(frame, config)


def validate_point_in_time_data(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Validate and normalize the canonical input schema."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], utc=False).dt.normalize()
    data["delist_date"] = pd.to_datetime(data["delist_date"], errors="coerce").dt.normalize()
    data["security_id"] = data["security_id"].astype(str)
    data["ticker"] = data["ticker"].astype(str)
    data["company_name"] = data["company_name"].astype(str)
    data["country"] = data["country"].astype(str)
    data["currency"] = data["currency"].astype(str)

    if "open" not in data.columns:
        data["open"] = np.nan

    numeric_columns = [
        "market_cap_usd",
        "open",
        "close",
        "adj_close",
        "fx_to_usd",
        "split_factor",
        "dividend",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.duplicated(["date", "security_id"]).any():
        dupes = data.loc[data.duplicated(["date", "security_id"], keep=False), ["date", "security_id"]]
        raise DataValidationError(f"Duplicate date/security_id rows found: {dupes.head().to_dict('records')}")

    required_positive = ["market_cap_usd", "close", "adj_close", "fx_to_usd"]
    for column in required_positive:
        bad = data[column].isna() | (data[column] <= 0)
        if bad.any():
            raise DataValidationError(f"Column {column} has missing or non-positive values")

    if data["split_factor"].isna().any() or (data["split_factor"] <= 0).any():
        raise DataValidationError("split_factor must be positive; use 1.0 when no split occurred")
    if data["dividend"].isna().any():
        raise DataValidationError("dividend must be present; use 0.0 when no dividend occurred")

    data["is_active"] = data["is_active"].map(_parse_bool)
    if data["is_active"].isna().any():
        raise DataValidationError("is_active must be boolean-like")

    if config.start_date:
        data = data[data["date"] >= pd.Timestamp(config.start_date)]
    if config.end_date:
        data = data[data["date"] <= pd.Timestamp(config.end_date)]

    data["open"] = data["open"].fillna(data["close"])
    data["price_close_usd"] = data["close"] * data["fx_to_usd"]
    data["price_open_usd"] = data["open"] * data["fx_to_usd"]
    data["adj_close_usd"] = data["adj_close"] * data["fx_to_usd"]
    data = data.sort_values(["date", "security_id"]).reset_index(drop=True)

    dates = data["date"].drop_duplicates()
    if len(dates) <= config.trigger_lookback_days + 1:
        raise DataValidationError(
            "Data has too few dates for the configured trigger_lookback_days"
        )
    min_members = (
        data[data["is_active"]]
        .groupby("date")["security_id"]
        .nunique()
        .reindex(dates, fill_value=0)
        .min()
    )
    if min_members < config.min_members_per_date:
        raise DataValidationError(
            f"At least one date has fewer active members than min_members_per_date={config.min_members_per_date}"
        )
    if config.data_source_kind not in {"point_in_time", "public_rankings_proxy"}:
        raise DataValidationError(
            "data.source_kind must be point_in_time or public_rankings_proxy"
        )
    return data


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None
