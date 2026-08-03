"""Vendor export importer for the hourly top-10 equity pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_intraday_deepm.config import project_path
from equity_intraday_deepm.data import (
    EquityDataError,
    build_universe,
    read_table,
    validate_source,
    write_table,
)
from equity_intraday_deepm.config import DataConfig


SOURCE_REQUIREMENTS = {
    "canonical": {
        "intraday": "LSEG Tick History or FactSet Tick History for global equities, hourly OHLCV built from trades, 2000-current.",
        "security_master": "FactSet/LSEG/Bloomberg point-in-time symbology, exchange, listing status, security type, country, sector, currency.",
        "market_caps": "Point-in-time daily USD market cap from prices and shares outstanding, or raw price/shares/FX fields to compute it.",
        "broker_mapping": "IBKR conId/listing mapping for every eligible security, including historical ticker changes.",
    },
    "not_correct_for_canonical": [
        "Yahoo/yfinance: hourly history is too short and no point-in-time global security master.",
        "Wikipedia rankings: only ranking snapshots, not daily/hourly market-cap membership.",
        "IBKR historical API alone: good for current trading and limited backfill, not efficient canonical 2000-current global research data.",
        "US-only providers: usable for a proxy, not global top-10 correctness.",
    ],
}


@dataclass(frozen=True)
class ImportConfig:
    security_master_input: Path
    market_caps_input: Path
    hourly_bars_input: Path
    security_master_output: Path
    market_caps_output: Path
    hourly_bars_output: Path
    column_maps: dict[str, dict[str, str]]
    start: str
    end: str
    active_top_n: int
    strict_top_n: bool


@dataclass(frozen=True)
class BuildConfig:
    security_master_input: Path
    daily_prices_input: Path
    daily_shares_input: Path
    daily_fx_input: Path | None
    hourly_bars_input: Path
    security_master_output: Path
    market_caps_output: Path
    hourly_bars_output: Path
    column_maps: dict[str, dict[str, str]]
    start: str
    end: str
    active_top_n: int
    strict_top_n: bool


def load_import_config(path: str | Path) -> ImportConfig:
    cfg_path = project_path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Import config must be a mapping: {cfg_path}")
    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})
    return ImportConfig(
        security_master_input=project_path(inputs["security_master"]),
        market_caps_input=project_path(inputs["market_caps"]),
        hourly_bars_input=project_path(inputs["hourly_bars"]),
        security_master_output=project_path(outputs.get("security_master", "equity_intraday_deepm/vendor_data/security_master.parquet")),
        market_caps_output=project_path(outputs.get("market_caps", "equity_intraday_deepm/vendor_data/market_caps.parquet")),
        hourly_bars_output=project_path(outputs.get("hourly_bars", "equity_intraday_deepm/vendor_data/hourly_bars.parquet")),
        column_maps=raw.get("column_maps", {}),
        start=str(raw.get("start", "2000-01-01")),
        end=str(raw.get("end", pd.Timestamp.today().date().isoformat())),
        active_top_n=int(raw.get("active_top_n", 10)),
        strict_top_n=bool(raw.get("strict_top_n", True)),
    )


def load_build_config(path: str | Path) -> BuildConfig:
    cfg_path = project_path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Build config must be a mapping: {cfg_path}")
    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})
    daily_fx = inputs.get("daily_fx")
    return BuildConfig(
        security_master_input=project_path(inputs["security_master"]),
        daily_prices_input=project_path(inputs["daily_prices"]),
        daily_shares_input=project_path(inputs["daily_shares"]),
        daily_fx_input=project_path(daily_fx) if daily_fx else None,
        hourly_bars_input=project_path(inputs["hourly_bars"]),
        security_master_output=project_path(outputs.get("security_master", "equity_intraday_deepm/vendor_data/security_master.parquet")),
        market_caps_output=project_path(outputs.get("market_caps", "equity_intraday_deepm/vendor_data/market_caps.parquet")),
        hourly_bars_output=project_path(outputs.get("hourly_bars", "equity_intraday_deepm/vendor_data/hourly_bars.parquet")),
        column_maps=raw.get("column_maps", {}),
        start=str(raw.get("start", "2000-01-01")),
        end=str(raw.get("end", pd.Timestamp.today().date().isoformat())),
        active_top_n=int(raw.get("active_top_n", 10)),
        strict_top_n=bool(raw.get("strict_top_n", True)),
    )


def import_vendor_exports(path: str | Path) -> dict[str, Any]:
    cfg = load_import_config(path)
    master = _standardize_security_master(read_table(cfg.security_master_input), cfg.column_maps.get("security_master", {}))
    caps = _standardize_market_caps(read_table(cfg.market_caps_input), cfg.column_maps.get("market_caps", {}))
    bars = _standardize_hourly_bars(read_table(cfg.hourly_bars_input), cfg.column_maps.get("hourly_bars", {}))

    write_table(master, cfg.security_master_output)
    write_table(caps, cfg.market_caps_output)
    write_table(bars, cfg.hourly_bars_output)

    data_cfg = DataConfig(
        active_top_n=cfg.active_top_n,
        security_master_path=cfg.security_master_output,
        market_caps_path=cfg.market_caps_output,
        hourly_bars_path=cfg.hourly_bars_output,
        universe_output_path=project_path("equity_intraday_deepm/data/top10_membership.parquet"),
        features_output_path=project_path("equity_intraday_deepm/data/features_hourly_top10.parquet"),
        audit_output_path=project_path("equity_intraday_deepm/data/vendor_import_audit.json"),
        source_label="Imported vendor point-in-time global equities",
        source_kind="vendor_point_in_time",
        commission_bps=0.5,
        half_spread_bps=1.0,
        slippage_bps=1.0,
        min_active_members=cfg.active_top_n,
    )
    source_summary = validate_source(data_cfg)
    universe_path, audit_path = build_universe(data_cfg, cfg.start, cfg.end, cfg.active_top_n)
    universe = pd.read_parquet(universe_path)
    active_counts = universe.groupby("timestamp")["security_id"].nunique()
    min_active = int(active_counts.min())
    if cfg.strict_top_n and min_active < cfg.active_top_n:
        raise EquityDataError(
            f"Imported data is not complete: minimum active membership is {min_active}, "
            f"expected {cfg.active_top_n}. See {audit_path}."
        )
    audit = {
        "outputs": {
            "security_master": str(cfg.security_master_output),
            "market_caps": str(cfg.market_caps_output),
            "hourly_bars": str(cfg.hourly_bars_output),
            "universe": str(universe_path),
        },
        "source_summary": source_summary,
        "active_top_n": cfg.active_top_n,
        "min_active_members": min_active,
        "strict_top_n": cfg.strict_top_n,
        "date_range": {"start": cfg.start, "end": cfg.end},
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def build_canonical_from_vendor_exports(path: str | Path) -> dict[str, Any]:
    cfg = load_build_config(path)
    master = _standardize_security_master(
        read_table(cfg.security_master_input),
        cfg.column_maps.get("security_master", {}),
    )
    prices = _standardize_daily_prices(
        read_table(cfg.daily_prices_input),
        cfg.column_maps.get("daily_prices", {}),
    )
    shares = _standardize_daily_shares(
        read_table(cfg.daily_shares_input),
        cfg.column_maps.get("daily_shares", {}),
    )
    fx = (
        _standardize_daily_fx(read_table(cfg.daily_fx_input), cfg.column_maps.get("daily_fx", {}))
        if cfg.daily_fx_input is not None
        else None
    )
    market_caps = _build_market_caps_from_components(prices, shares, fx)
    hourly = _standardize_hourly_bars(
        read_table(cfg.hourly_bars_input),
        cfg.column_maps.get("hourly_bars", {}),
    )
    if fx is not None and "currency" in hourly.columns and hourly["fx_to_usd"].isna().any():
        hourly = _attach_fx_to_hourly(hourly, fx)

    write_table(master, cfg.security_master_output)
    write_table(market_caps, cfg.market_caps_output)
    write_table(hourly, cfg.hourly_bars_output)

    import_cfg = {
        "start": cfg.start,
        "end": cfg.end,
        "active_top_n": cfg.active_top_n,
        "strict_top_n": cfg.strict_top_n,
        "inputs": {
            "security_master": str(cfg.security_master_output),
            "market_caps": str(cfg.market_caps_output),
            "hourly_bars": str(cfg.hourly_bars_output),
        },
        "outputs": {
            "security_master": str(cfg.security_master_output),
            "market_caps": str(cfg.market_caps_output),
            "hourly_bars": str(cfg.hourly_bars_output),
        },
    }
    temp_config = cfg.security_master_output.parent / "_canonical_import_config.json"
    temp_config.write_text(json.dumps(import_cfg, indent=2), encoding="utf-8")
    try:
        audit = import_vendor_exports(temp_config)
    finally:
        temp_config.unlink(missing_ok=True)
    audit["component_rows"] = {
        "daily_prices": int(len(prices)),
        "daily_shares": int(len(shares)),
        "daily_fx": int(len(fx)) if fx is not None else 0,
        "market_caps_built": int(len(market_caps)),
    }
    audit_path = project_path("equity_intraday_deepm/data/vendor_build_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    audit["outputs"]["build_audit"] = str(audit_path)
    return audit


def _standardize_security_master(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    data = _rename(frame, column_map)
    required = [
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
    ]
    _require(data, required, "security_master")
    optional = ["company_id", "primary_security_id", "is_primary_listing", "share_class"]
    keep_cols = required + [col for col in optional if col in data.columns]
    data = data[keep_cols].copy()
    if "company_id" not in data.columns:
        data["company_id"] = data["security_id"]
    if "primary_security_id" not in data.columns:
        data["primary_security_id"] = data["security_id"]
    if "is_primary_listing" not in data.columns:
        data["is_primary_listing"] = data["security_id"] == data["primary_security_id"]
    data["listing_start"] = pd.to_datetime(data["listing_start"], errors="coerce")
    data["listing_end"] = pd.to_datetime(data["listing_end"], errors="coerce")
    data = data.drop_duplicates("security_id", keep="last")
    return data


def _standardize_market_caps(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    data = _rename(frame, column_map)
    if "market_cap_usd" not in data.columns:
        if {"shares_outstanding", "close", "fx_to_usd"}.issubset(data.columns):
            data["market_cap_usd"] = (
                pd.to_numeric(data["shares_outstanding"], errors="coerce")
                * pd.to_numeric(data["close"], errors="coerce")
                * pd.to_numeric(data["fx_to_usd"], errors="coerce")
            )
        else:
            raise EquityDataError(
                "market_caps needs market_cap_usd or shares_outstanding + close + fx_to_usd"
            )
    _require(data, ["date", "security_id", "market_cap_usd"], "market_caps")
    data = data[["date", "security_id", "market_cap_usd"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["market_cap_usd"] = pd.to_numeric(data["market_cap_usd"], errors="coerce")
    data = data.dropna(subset=["date", "security_id", "market_cap_usd"])
    data = data[data["market_cap_usd"] > 0]
    data = data.drop_duplicates(["date", "security_id"], keep="last")
    return data


def _standardize_hourly_bars(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    data = _rename(frame, column_map)
    _require(data, ["timestamp", "security_id", "open", "high", "low", "close", "volume"], "hourly_bars")
    has_fx = "fx_to_usd" in data.columns
    if not has_fx:
        data["fx_to_usd"] = pd.NA
    if "adj_close" not in data.columns:
        data["adj_close"] = data["close"]
    cols = ["timestamp", "security_id", "open", "high", "low", "close", "adj_close", "volume", "fx_to_usd"]
    if "currency" in data.columns:
        cols.append("currency")
    data = data[cols].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "adj_close", "volume", "fx_to_usd"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if not has_fx and "currency" not in data.columns:
        data["fx_to_usd"] = 1.0
    if "currency" in data.columns:
        data.loc[data["currency"].astype(str).eq("USD") & data["fx_to_usd"].isna(), "fx_to_usd"] = 1.0
    required_non_null = ["timestamp", "security_id", "close", "adj_close"]
    if "currency" not in data.columns:
        required_non_null.append("fx_to_usd")
    data = data.dropna(subset=required_non_null)
    data = data.drop_duplicates(["timestamp", "security_id"], keep="last")
    return data


def _standardize_daily_prices(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    data = _rename(frame, column_map)
    _require(data, ["date", "security_id", "close"], "daily_prices")
    if "currency" not in data.columns:
        data["currency"] = "USD"
    if "fx_to_usd" not in data.columns:
        data["fx_to_usd"] = pd.NA
    data = data[["date", "security_id", "close", "currency", "fx_to_usd"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["fx_to_usd"] = pd.to_numeric(data["fx_to_usd"], errors="coerce")
    return data.dropna(subset=["date", "security_id", "close"])


def _standardize_daily_shares(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    data = _rename(frame, column_map)
    _require(data, ["date", "security_id", "shares_outstanding"], "daily_shares")
    data = data[["date", "security_id", "shares_outstanding"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["shares_outstanding"] = pd.to_numeric(data["shares_outstanding"], errors="coerce")
    return data.dropna(subset=["date", "security_id", "shares_outstanding"])


def _standardize_daily_fx(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    data = _rename(frame, column_map)
    _require(data, ["date", "currency", "fx_to_usd"], "daily_fx")
    data = data[["date", "currency", "fx_to_usd"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["currency"] = data["currency"].astype(str)
    data["fx_to_usd"] = pd.to_numeric(data["fx_to_usd"], errors="coerce")
    return data.dropna(subset=["date", "currency", "fx_to_usd"]).drop_duplicates(["date", "currency"], keep="last")


def _build_market_caps_from_components(
    prices: pd.DataFrame,
    shares: pd.DataFrame,
    fx: pd.DataFrame | None,
) -> pd.DataFrame:
    merged_parts = []
    shares_by_security = {
        security_id: group.sort_values("date")
        for security_id, group in shares.groupby("security_id", sort=False)
    }
    for security_id, price_group in prices.groupby("security_id", sort=False):
        share_group = shares_by_security.get(security_id)
        if share_group is None or share_group.empty:
            continue
        merged_parts.append(
            pd.merge_asof(
                price_group.sort_values("date"),
                share_group[["date", "shares_outstanding"]].sort_values("date"),
                on="date",
                direction="backward",
                allow_exact_matches=True,
            ).assign(security_id=security_id)
        )
    if not merged_parts:
        raise EquityDataError("Could not match any daily prices to point-in-time shares outstanding")
    merged = pd.concat(merged_parts, ignore_index=True)
    if fx is not None:
        needs_fx = merged["fx_to_usd"].isna()
        if needs_fx.any():
            merged = merged.merge(fx, on=["date", "currency"], how="left", suffixes=("", "_from_table"))
            merged["fx_to_usd"] = merged["fx_to_usd"].fillna(merged["fx_to_usd_from_table"])
            merged = merged.drop(columns=["fx_to_usd_from_table"], errors="ignore")
    merged.loc[merged["currency"].eq("USD") & merged["fx_to_usd"].isna(), "fx_to_usd"] = 1.0
    merged = merged.dropna(subset=["close", "shares_outstanding", "fx_to_usd"])
    merged["market_cap_usd"] = merged["close"] * merged["shares_outstanding"] * merged["fx_to_usd"]
    out = merged[["date", "security_id", "market_cap_usd"]].copy()
    out = out[out["market_cap_usd"] > 0]
    return out.drop_duplicates(["date", "security_id"], keep="last")


def _attach_fx_to_hourly(hourly: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    data["date"] = pd.to_datetime(data["timestamp"]).dt.normalize()
    parts = []
    fx_by_currency = {
        currency: group.sort_values("date")
        for currency, group in fx.groupby("currency", sort=False)
    }
    for currency, group in data.groupby("currency", sort=False):
        group = group.sort_values("date")
        if str(currency) == "USD":
            group["fx_to_usd"] = group["fx_to_usd"].fillna(1.0)
            parts.append(group)
            continue
        fx_group = fx_by_currency.get(currency)
        if fx_group is None or fx_group.empty:
            parts.append(group)
            continue
        merged = pd.merge_asof(
            group,
            fx_group[["date", "fx_to_usd"]].sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", "_from_table"),
        )
        merged["fx_to_usd"] = merged["fx_to_usd"].fillna(merged["fx_to_usd_from_table"])
        parts.append(merged.drop(columns=["fx_to_usd_from_table"], errors="ignore"))
    return pd.concat(parts, ignore_index=True).drop(columns=["date"], errors="ignore")


def _rename(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    inverse = {vendor_col: canonical_col for canonical_col, vendor_col in column_map.items()}
    return frame.rename(columns=inverse).copy()


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise EquityDataError(f"{name} missing required columns after mapping: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import vendor exports for hourly top-10 equity DeePM")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources")
    imp = sub.add_parser("import-vendor")
    imp.add_argument("--config", required=True)
    build = sub.add_parser("build-canonical")
    build.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "sources":
        print(json.dumps(SOURCE_REQUIREMENTS, indent=2, sort_keys=True))
    elif args.command == "import-vendor":
        try:
            print(json.dumps(import_vendor_exports(args.config), indent=2, sort_keys=True))
        except EquityDataError as exc:
            print(f"Equity data error: {exc}")
            raise SystemExit(1) from exc
    elif args.command == "build-canonical":
        try:
            print(json.dumps(build_canonical_from_vendor_exports(args.config), indent=2, sort_keys=True))
        except EquityDataError as exc:
            print(f"Equity data error: {exc}")
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
