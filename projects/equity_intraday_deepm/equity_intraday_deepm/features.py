"""Hourly feature construction for the top-N equity pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_intraday_deepm.config import DataConfig
from equity_intraday_deepm.data import read_table, write_table


FEATURE_COLUMNS = [
    "r1h",
    "r3h",
    "r6h",
    "r1d",
    "r3d",
    "r1w",
    "z_price_1d",
    "z_price_1w",
    "realized_vol_1d",
    "realized_vol_1w",
    "intraday_range",
    "z_volume_1w",
    "rank_scaled",
    "rank_change",
    "market_cap_log",
    "cpd_short",
    "cpd_medium",
]


def build_features(config: DataConfig) -> tuple[Path, Path]:
    bars = read_table(config.hourly_bars_path).copy()
    universe = read_table(config.universe_output_path).copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    universe["timestamp"] = pd.to_datetime(universe["timestamp"])
    bars["security_id"] = bars["security_id"].astype(str)
    universe["security_id"] = universe["security_id"].astype(str)

    ever_active = sorted(universe["security_id"].unique())
    bars = bars[bars["security_id"].isin(ever_active)].sort_values(["security_id", "timestamp"]).copy()
    bars["price_usd"] = bars["adj_close"].astype(float) * bars["fx_to_usd"].astype(float)
    bars["close_usd"] = bars["close"].astype(float) * bars["fx_to_usd"].astype(float)

    frames = []
    for _, group in bars.groupby("security_id", sort=False):
        group = group.sort_values("timestamp").copy()
        price = group["price_usd"]
        group["r1h"] = price.pct_change(1)
        group["r3h"] = price.pct_change(3)
        group["r6h"] = price.pct_change(6)
        group["r1d"] = price.pct_change(6)
        group["r3d"] = price.pct_change(18)
        group["r1w"] = price.pct_change(30)
        logp = np.log(price)
        group["z_price_1d"] = _rolling_z(logp, 6)
        group["z_price_1w"] = _rolling_z(logp, 30)
        group["realized_vol_1d"] = group["r1h"].rolling(6, min_periods=3).std(ddof=0)
        group["realized_vol_1w"] = group["r1h"].rolling(30, min_periods=6).std(ddof=0)
        group["intraday_range"] = (group["high"].astype(float) - group["low"].astype(float)) / group["close"].astype(float)
        group["z_volume_1w"] = _rolling_z(np.log1p(group["volume"].astype(float)), 30)
        vol_short = group["r1h"].rolling(12, min_periods=4).std(ddof=0)
        vol_medium = group["r1h"].rolling(48, min_periods=8).std(ddof=0)
        group["cpd_short"] = group["r1h"].abs() / (vol_short + 1e-8)
        group["cpd_medium"] = group["r6h"].abs() / (vol_medium + 1e-8)
        group["target"] = price.shift(-1) / price - 1.0
        frames.append(group)

    features = pd.concat(frames, ignore_index=True)
    active_cols = [
        "timestamp",
        "company_id",
        "security_id",
        "ticker",
        "company_name",
        "country",
        "sector",
        "currency",
        "exchange",
        "ibkr_con_id",
        "market_cap_usd",
        "security_market_cap_usd",
        "rank",
        "is_active",
    ]
    active_cols = [col for col in active_cols if col in universe.columns]
    active = universe[active_cols].copy()
    features = features.merge(active, on=["timestamp", "security_id"], how="left")

    latest_meta = (
        universe.sort_values("timestamp")
        .drop_duplicates("security_id", keep="last")
        .set_index("security_id")
    )
    for col in ["ticker", "company_name", "country", "sector", "currency", "exchange", "ibkr_con_id"]:
        features[col] = features[col].fillna(features["security_id"].map(latest_meta[col]))
    features["is_active"] = features["is_active"].fillna(False).astype(bool)
    features["rank"] = pd.to_numeric(features["rank"], errors="coerce")
    features["market_cap_usd"] = pd.to_numeric(features["market_cap_usd"], errors="coerce")
    features["market_cap_usd"] = features["market_cap_usd"].fillna(
        features["security_id"].map(latest_meta["market_cap_usd"])
    )
    features["rank_scaled"] = (config.active_top_n + 1 - features["rank"].fillna(config.active_top_n + 1)) / config.active_top_n
    features["rank_change"] = (
        features.sort_values(["security_id", "timestamp"])
        .groupby("security_id")["rank"]
        .diff()
        .fillna(0.0)
    )
    features["market_cap_log"] = np.log(features["market_cap_usd"].clip(lower=1.0))
    features["transaction_cost_bps"] = (
        config.commission_bps + 2.0 * config.half_spread_bps + config.slippage_bps
    )
    features = features.sort_values(["timestamp", "security_id"]).reset_index(drop=True)
    write_table(features, config.features_output_path)

    active_counts = features[features["is_active"]].groupby("timestamp")["security_id"].nunique()
    audit = {
        "features_path": str(config.features_output_path),
        "rows": int(len(features)),
        "timestamps": int(features["timestamp"].nunique()),
        "securities": int(features["security_id"].nunique()),
        "active_members_per_timestamp": active_counts.describe().to_dict(),
        "features": FEATURE_COLUMNS,
        "target": "next-hour adjusted USD return",
        "leakage_policy": "features use current and prior hourly bars; target is shifted after feature construction",
    }
    config.audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.audit_output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return config.features_output_path, config.audit_output_path


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 4)).mean()
    std = series.rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
    return (series - mean) / (std + 1e-8)
