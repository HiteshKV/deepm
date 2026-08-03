"""Live EOD snapshot persistence for daily inference."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from deepm.live.config import path_from_config


def snapshot_dir(config: Mapping[str, object]) -> Path:
    """Return the configured live snapshot directory."""
    directory = path_from_config(config, "data", "live_snapshot_dir")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def latest_snapshot_path(config: Mapping[str, object]) -> Path:
    """Return the stable live snapshot parquet path."""
    data_cfg = config.get("data", {})
    filename = str(data_cfg.get("latest_snapshot_name", "latest_prices.parquet"))
    return snapshot_dir(config) / filename


def dated_snapshot_path(config: Mapping[str, object], run_date: date) -> Path:
    """Return the immutable dated live snapshot parquet path."""
    return snapshot_dir(config) / f"prices_{run_date.strftime('%Y%m%d')}.parquet"


def persist_live_snapshot(
    raw_prices: pd.DataFrame,
    config: Mapping[str, object],
    run_date: date,
    provider_name: str,
) -> Path:
    """Persist an immutable and latest live price snapshot.

    This intentionally writes under ``live_state`` rather than mutating the
    training raw-data parquet.
    """
    data = raw_prices.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    dated_path = dated_snapshot_path(config, run_date)
    latest_path = latest_snapshot_path(config)
    data.to_parquet(dated_path)
    data.to_parquet(latest_path)

    audit = {
        "provider": provider_name,
        "run_date": run_date.isoformat(),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "first_date": None if data.empty else str(pd.Timestamp(data.index.min()).date()),
        "last_date": None if data.empty else str(pd.Timestamp(data.index.max()).date()),
        "dated_snapshot": str(dated_path),
        "latest_snapshot": str(latest_path),
    }
    (snapshot_dir(config) / "latest_snapshot.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return latest_path
