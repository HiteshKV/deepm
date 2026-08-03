"""Feature preparation for daily live runs without mutating training data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from deepm.data.build_features import add_cross_sectional_features, prepare_features
from deepm.live.exceptions import DataUnavailable


def derive_feature_cache_path(raw_price_parquet: Path) -> Path:
    """Return the standard feature-cache path for a raw parquet snapshot."""
    return raw_price_parquet.with_name(f"feats-{raw_price_parquet.name}")


def load_or_build_features(
    raw_prices: pd.DataFrame,
    raw_price_parquet: Path,
    run_date: date,
    prefer_cache: bool = True,
) -> pd.DataFrame:
    """Load an existing feature cache when current enough, else build in memory."""
    cache = derive_feature_cache_path(raw_price_parquet)
    if prefer_cache and cache.exists():
        features = pd.read_parquet(cache)
        features.index = pd.to_datetime(features.index)
        features = features.sort_index()
        if features.index.max() >= pd.Timestamp(run_date):
            return features.loc[: pd.Timestamp(run_date)].copy()

    if raw_prices.empty:
        raise DataUnavailable("Cannot build features from an empty raw price panel")
    features = prepare_features(raw_prices)
    features = features.drop(columns="date", errors="ignore")
    features = add_cross_sectional_features(features)
    features.index = pd.to_datetime(features.index)
    features = features.sort_index()
    features = features.loc[: pd.Timestamp(run_date)].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache)
    return features


def latest_feature_rows(features: pd.DataFrame, run_date: date) -> pd.DataFrame:
    """Return the latest available feature row for each ticker up to run_date."""
    if "ticker" not in features.columns:
        raise DataUnavailable("Feature data must include a ticker column")
    eligible = features.loc[features.index <= pd.Timestamp(run_date)].copy()
    if eligible.empty:
        raise DataUnavailable(f"No feature rows available on or before {run_date.isoformat()}")
    latest = eligible.sort_index().groupby("ticker", as_index=False).tail(1)
    latest = latest.set_index("ticker", drop=False)
    return latest


def feature_freshness_check(
    features: pd.DataFrame,
    run_date: date,
    max_staleness_days: int,
) -> tuple[bool, int]:
    """Check whether the latest feature date is within the allowed staleness."""
    if features.empty:
        return False, 999999
    latest = pd.Timestamp(features.index.max()).date()
    stale_days = (run_date - latest).days
    return stale_days <= max_staleness_days, stale_days
