"""Vendor-schema validation and causal top-N universe construction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from equity_intraday_deepm.config import DataConfig, load_data_config


MASTER_COLUMNS = {
    "security_id",
    "ticker",
    "company_name",
    "country",
    "sector",
    "currency",
    "exchange",
    "ibkr_con_id",
    "is_common_equity",
    "is_ibkr_tradable",
    "listing_start",
    "listing_end",
}
OPTIONAL_MASTER_COLUMNS = {
    "company_id",
    "primary_security_id",
    "is_primary_listing",
    "share_class",
}
CAP_COLUMNS = {"date", "security_id", "market_cap_usd"}
BAR_COLUMNS = {
    "timestamp",
    "security_id",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "fx_to_usd",
}


class EquityDataError(ValueError):
    """Raised when hourly equity source data is missing or invalid."""


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise EquityDataError(f"Required source file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise EquityDataError(f"Unsupported file type for {path}; use CSV or Parquet")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def validate_source(config: DataConfig) -> dict[str, object]:
    master = read_table(config.security_master_path)
    caps = read_table(config.market_caps_path)
    bars = read_table(config.hourly_bars_path)
    _require_columns(master, MASTER_COLUMNS, "security_master")
    _require_columns(caps, CAP_COLUMNS, "market_caps")
    _require_columns(bars, BAR_COLUMNS, "hourly_bars")

    master = _normalise_master(master)
    caps = _normalise_caps(caps)
    bars = _normalise_bars(bars)
    duplicate_caps = int(caps.duplicated(["date", "security_id"]).sum())
    duplicate_bars = int(bars.duplicated(["timestamp", "security_id"]).sum())
    if duplicate_caps:
        raise EquityDataError(f"market_caps has {duplicate_caps} duplicate date/security rows")
    if duplicate_bars:
        raise EquityDataError(f"hourly_bars has {duplicate_bars} duplicate timestamp/security rows")

    tradable = master[master["is_common_equity"] & master["is_ibkr_tradable"]]
    if tradable.empty:
        raise EquityDataError("No common-equity IBKR-tradable securities found")
    return {
        "security_master_rows": int(len(master)),
        "market_cap_rows": int(len(caps)),
        "hourly_bar_rows": int(len(bars)),
        "tradable_common_equities": int(len(tradable)),
        "first_bar": bars["timestamp"].min().isoformat(),
        "last_bar": bars["timestamp"].max().isoformat(),
    }


def build_universe(
    config: DataConfig,
    start: str,
    end: str,
    active_top_n: int | None = None,
) -> tuple[Path, Path]:
    active_top_n = int(active_top_n or config.active_top_n)
    master = _normalise_master(read_table(config.security_master_path))
    caps = _normalise_caps(read_table(config.market_caps_path))
    bars = _normalise_bars(read_table(config.hourly_bars_path))

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    bars = bars[(bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)].copy()
    if bars.empty:
        raise EquityDataError("No hourly bars in requested date range")

    eligible_master = master[master["is_common_equity"] & master["is_ibkr_tradable"]].copy()
    eligible_ids = set(eligible_master["security_id"])
    caps = caps[caps["security_id"].isin(eligible_ids)].sort_values(["date", "market_cap_usd"])
    cap_by_date = {date: frame for date, frame in caps.groupby("date", sort=True)}
    cap_dates = np.asarray(sorted(cap_by_date), dtype="datetime64[ns]")
    timestamps = pd.Index(sorted(bars["timestamp"].drop_duplicates()))

    rows: list[dict[str, object]] = []
    skipped_no_cap = 0
    insufficient_candidates = 0
    missing_active_bars = 0
    for ts in timestamps:
        prior_cap_date = _prior_cap_date(ts, cap_dates)
        if prior_cap_date is None:
            skipped_no_cap += 1
            continue
        cap_frame = cap_by_date[pd.Timestamp(prior_cap_date)].copy()
        cap_frame = cap_frame[cap_frame["security_id"].isin(eligible_ids)]
        cap_frame = cap_frame.merge(eligible_master, on="security_id", how="inner")
        active_listing = (
            (cap_frame["listing_start"].isna() | (cap_frame["listing_start"] <= ts))
            & (cap_frame["listing_end"].isna() | (cap_frame["listing_end"] >= ts))
        )
        cap_frame = cap_frame[active_listing].copy()
        cap_frame = cap_frame.rename(columns={"market_cap_usd": "security_market_cap_usd"})
        company_caps = (
            cap_frame.groupby("company_id", as_index=False)["security_market_cap_usd"]
            .sum()
            .rename(columns={"security_market_cap_usd": "market_cap_usd"})
            .sort_values("market_cap_usd", ascending=False)
        )
        top_companies = company_caps.head(active_top_n)
        if len(top_companies) < active_top_n:
            insufficient_candidates += 1
        bar_ids = set(bars.loc[bars["timestamp"] == ts, "security_id"])
        selected_rows = []
        for company_row in top_companies.itertuples(index=False):
            candidates = cap_frame[cap_frame["company_id"] == company_row.company_id].copy()
            candidates = candidates[candidates["security_id"].isin(bar_ids)]
            if candidates.empty:
                continue
            candidates["_primary_rank"] = candidates["is_primary_listing"].astype(int)
            candidates = candidates.sort_values(
                ["_primary_rank", "security_market_cap_usd"],
                ascending=[False, False],
            )
            selected = candidates.iloc[0].copy()
            selected["market_cap_usd"] = float(company_row.market_cap_usd)
            selected_rows.append(selected)
        if len(selected_rows) < active_top_n:
            missing_active_bars += 1
        for rank, row in enumerate(selected_rows, start=1):
            rows.append(
                {
                    "timestamp": ts,
                    "company_id": row["company_id"],
                    "security_id": row["security_id"],
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "country": row["country"],
                    "sector": row["sector"],
                    "currency": row["currency"],
                    "exchange": row["exchange"],
                    "ibkr_con_id": row["ibkr_con_id"],
                    "market_cap_usd": float(row["market_cap_usd"]),
                    "security_market_cap_usd": float(row["security_market_cap_usd"]),
                    "rank": rank,
                    "is_active": True,
                    "market_cap_date": pd.Timestamp(prior_cap_date),
                }
            )

    universe = pd.DataFrame(rows).sort_values(["timestamp", "rank", "security_id"])
    if universe.empty:
        raise EquityDataError("No active top-N membership rows could be built")
    write_table(universe, config.universe_output_path)
    active_counts = universe.groupby("timestamp")["security_id"].nunique()
    audit = {
        "config": asdict(config),
        "active_top_n": active_top_n,
        "source_label": config.source_label,
        "source_kind": config.source_kind,
        "first_timestamp": universe["timestamp"].min().isoformat(),
        "last_timestamp": universe["timestamp"].max().isoformat(),
        "membership_rows": int(len(universe)),
        "unique_active_securities": int(universe["security_id"].nunique()),
        "unique_active_companies": int(universe["company_id"].nunique()) if "company_id" in universe else int(universe["security_id"].nunique()),
        "active_members_per_timestamp": active_counts.describe().to_dict(),
        "timestamps_without_prior_cap": skipped_no_cap,
        "timestamps_with_insufficient_candidates": insufficient_candidates,
        "timestamps_with_missing_active_bars": missing_active_bars,
        "causal_rule": "membership uses company-level market cap from the latest market_cap date strictly before the bar date",
    }
    _write_json(config.audit_output_path, audit)
    return config.universe_output_path, config.audit_output_path


def _prior_cap_date(ts: pd.Timestamp, cap_dates: np.ndarray) -> pd.Timestamp | None:
    cutoff = np.datetime64(pd.Timestamp(ts).normalize())
    idx = int(np.searchsorted(cap_dates, cutoff, side="left")) - 1
    if idx < 0:
        return None
    return pd.Timestamp(cap_dates[idx])


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise EquityDataError(f"{name} missing required columns: {missing}")


def _normalise_master(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, MASTER_COLUMNS, "security_master")
    out = frame.copy()
    for col in ["security_id", "ticker", "company_name", "country", "sector", "currency", "exchange"]:
        out[col] = out[col].astype(str)
    out["ibkr_con_id"] = out["ibkr_con_id"].astype(str)
    out["is_common_equity"] = out["is_common_equity"].map(_parse_bool)
    out["is_ibkr_tradable"] = out["is_ibkr_tradable"].map(_parse_bool)
    out["listing_start"] = pd.to_datetime(out["listing_start"], errors="coerce")
    out["listing_end"] = pd.to_datetime(out["listing_end"], errors="coerce")
    if "company_id" not in out.columns:
        out["company_id"] = out["security_id"]
    if "primary_security_id" not in out.columns:
        out["primary_security_id"] = out["security_id"]
    if "is_primary_listing" not in out.columns:
        out["is_primary_listing"] = out["security_id"] == out["primary_security_id"]
    out["company_id"] = out["company_id"].astype(str)
    out["primary_security_id"] = out["primary_security_id"].astype(str)
    out["is_primary_listing"] = out["is_primary_listing"].map(_parse_bool)
    out["is_primary_listing"] = out["is_primary_listing"].fillna(False)
    if out["is_common_equity"].isna().any() or out["is_ibkr_tradable"].isna().any():
        raise EquityDataError("is_common_equity and is_ibkr_tradable must be boolean-like")
    return out.drop_duplicates("security_id")


def _normalise_caps(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, CAP_COLUMNS, "market_caps")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["security_id"] = out["security_id"].astype(str)
    out["market_cap_usd"] = pd.to_numeric(out["market_cap_usd"], errors="coerce")
    out = out.dropna(subset=["date", "security_id", "market_cap_usd"])
    out = out[out["market_cap_usd"] > 0]
    return out


def _normalise_bars(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, BAR_COLUMNS, "hourly_bars")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out["security_id"] = out["security_id"].astype(str)
    for col in ["open", "high", "low", "close", "adj_close", "volume", "fx_to_usd"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "security_id", "close", "adj_close", "fx_to_usd"])
    out = out[(out["close"] > 0) & (out["adj_close"] > 0) & (out["fx_to_usd"] > 0)]
    return out.sort_values(["timestamp", "security_id"])


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True), encoding="utf-8")


def data_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hourly equity data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate-source")
    val.add_argument("--config", required=True)
    uni = sub.add_parser("build-universe")
    uni.add_argument("--config", required=True)
    uni.add_argument("--start", required=True)
    uni.add_argument("--end", required=True)
    uni.add_argument("--active-top-n", type=int, default=None)
    feat = sub.add_parser("build-features")
    feat.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_data_config(args.config)
    try:
        if args.command == "validate-source":
            print(json.dumps(validate_source(config), indent=2, sort_keys=True))
        elif args.command == "build-universe":
            out, audit = build_universe(config, args.start, args.end, args.active_top_n)
            print(f"universe={out}")
            print(f"audit={audit}")
        elif args.command == "build-features":
            from equity_intraday_deepm.features import build_features

            out, audit = build_features(config)
            print(f"features={out}")
            print(f"audit={audit}")
    except EquityDataError as exc:
        print(f"Equity data error: {exc}")
        raise SystemExit(1) from exc
