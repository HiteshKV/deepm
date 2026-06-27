"""Validate a raw DeePM price snapshot and any existing feature cache.

This script is intentionally read-only. It checks that the dated Yahoo proxy
snapshot has the expected 50-ticker wide price shape and, when the matching
``feats-`` parquet already exists, that the configured production features are
present for the same ticker universe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_EXPECTED_TICKERS = 50
DEFAULT_START_DATE = "1990-01-02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Raw wide price parquet to validate.")
    parser.add_argument(
        "--expected-tickers",
        type=int,
        default=DEFAULT_EXPECTED_TICKERS,
        help="Expected number of ticker columns in the raw snapshot.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Minimum required first snapshot date.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_settings/deepm-gat.yaml"),
        help="Training config whose ticker_subset/features should be checked.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def validate_raw(snapshot: Path, expected_tickers: int, start_date: str) -> pd.DataFrame:
    if not snapshot.exists():
        fail(f"snapshot not found: {snapshot}")

    data = pd.read_parquet(snapshot)
    if not isinstance(data.index, pd.DatetimeIndex):
        fail("raw snapshot index must be a DatetimeIndex")
    if data.empty:
        fail("raw snapshot is empty")
    if data.shape[1] != expected_tickers:
        fail(f"expected {expected_tickers} tickers, found {data.shape[1]}")
    if data.columns.duplicated().any():
        fail("raw snapshot has duplicate ticker columns")
    if data.index.has_duplicates:
        fail("raw snapshot has duplicate dates")
    if not data.index.is_monotonic_increasing:
        fail("raw snapshot dates must be sorted")

    required_start = pd.Timestamp(start_date)
    if data.index.min() > required_start:
        fail(f"first date {data.index.min().date()} is after required {required_start.date()}")
    today = pd.Timestamp.today().normalize()
    if data.index.max() > today:
        fail(f"last date {data.index.max().date()} is in the future")

    fully_missing = data.columns[data.isna().all()].tolist()
    if fully_missing:
        fail(f"tickers with no observations: {fully_missing}")

    missing_counts = data.notna().sum()
    if (missing_counts == 0).any():
        fail("one or more tickers have zero non-null rows")

    print(
        "Raw snapshot OK:",
        f"{data.shape[1]} tickers,",
        f"{len(data):,} dates,",
        f"{data.index.min().date()} to {data.index.max().date()}",
    )
    return data


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        fail(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(raw: pd.DataFrame, config_path: Path) -> dict:
    config = load_config(config_path)
    ticker_subset = config.get("ticker_subset", [])
    features = config.get("features", [])
    missing_tickers = sorted(set(ticker_subset) - set(raw.columns))
    if missing_tickers:
        fail(f"configured tickers missing from raw snapshot: {missing_tickers}")
    if len(ticker_subset) != raw.shape[1]:
        fail(
            f"config ticker_subset has {len(ticker_subset)} tickers, "
            f"raw snapshot has {raw.shape[1]}"
        )
    if not features:
        fail("config has no training features")
    print(f"Config OK: {len(ticker_subset)} tickers, {len(features)} training features")
    return config


def validate_feature_cache(snapshot: Path, raw: pd.DataFrame, config: dict) -> None:
    feats_path = snapshot.with_name(f"feats-{snapshot.name}")
    if not feats_path.exists():
        print(f"Feature cache not found yet: {feats_path} (run prepare_features next)")
        return

    features = pd.read_parquet(feats_path)
    required_columns = {"ticker", "close", "target", *config["features"]}
    missing_columns = sorted(required_columns - set(features.columns))
    if missing_columns:
        fail(f"feature cache missing columns: {missing_columns}")

    feature_tickers = set(features["ticker"].dropna().astype(str).unique())
    missing_feature_tickers = sorted(set(raw.columns) - feature_tickers)
    if missing_feature_tickers:
        fail(f"feature cache missing tickers: {missing_feature_tickers}")
    if features.empty:
        fail("feature cache is empty")

    print(
        "Feature cache OK:",
        f"{len(feature_tickers)} tickers,",
        f"{len(features):,} rows,",
        f"{feats_path}",
    )


def main() -> None:
    args = parse_args()
    raw = validate_raw(args.snapshot, args.expected_tickers, args.start_date)
    config = validate_config(raw, args.config)
    validate_feature_cache(args.snapshot, raw, config)


if __name__ == "__main__":
    main()
