"""Configuration loading for the top-10 stock strategy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StrategyConfig:
    run_name: str
    data_path: Path
    output_dir: Path
    data_source_label: str
    data_source_kind: str
    min_members_per_date: int
    start_date: str | None
    end_date: str | None
    initial_cash: float
    buy_notional: float
    trigger_return: float
    trigger_lookback_days: int
    top_n: int
    rebalance_frequency: str
    exit_ween_days: int
    commission_bps: float
    min_commission: float
    half_spread_bps: float
    slippage_bps: float
    max_position_pct_nav: float
    max_single_day_turnover_pct_nav: float
    allow_margin: bool
    allow_fractional_shares: bool
    risk_free_rate: float


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> StrategyConfig:
    """Load a YAML strategy config."""
    config_path = _project_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    data = raw.get("data", {})
    strategy = raw.get("strategy", {})
    execution = raw.get("execution", {})
    risk = raw.get("risk", {})
    reporting = raw.get("reporting", {})

    def required(section: dict[str, Any], key: str) -> Any:
        if key not in section:
            raise ValueError(f"Missing config key: {key}")
        return section[key]

    loaded = StrategyConfig(
        run_name=str(raw.get("run_name", "top10_buy_the_move")),
        data_path=_project_path(required(data, "path")),
        output_dir=_project_path(reporting.get("output_dir", "stock_top10_strategy/runs")),
        data_source_label=str(data.get("source_label", "unknown")),
        data_source_kind=str(data.get("source_kind", "point_in_time")),
        min_members_per_date=int(data.get("min_members_per_date", strategy.get("top_n", 10))),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
        initial_cash=float(required(strategy, "initial_cash")),
        buy_notional=float(required(strategy, "buy_notional")),
        trigger_return=float(required(strategy, "trigger_return")),
        trigger_lookback_days=int(required(strategy, "trigger_lookback_days")),
        top_n=int(required(strategy, "top_n")),
        rebalance_frequency=str(strategy.get("rebalance_frequency", "daily")),
        exit_ween_days=int(required(strategy, "exit_ween_days")),
        commission_bps=float(required(execution, "commission_bps")),
        min_commission=float(required(execution, "min_commission")),
        half_spread_bps=float(required(execution, "half_spread_bps")),
        slippage_bps=float(required(execution, "slippage_bps")),
        max_position_pct_nav=float(required(risk, "max_position_pct_nav")),
        max_single_day_turnover_pct_nav=float(required(risk, "max_single_day_turnover_pct_nav")),
        allow_margin=bool(risk.get("allow_margin", False)),
        allow_fractional_shares=bool(risk.get("allow_fractional_shares", False)),
        risk_free_rate=float(risk.get("risk_free_rate", 0.0)),
    )
    if loaded.rebalance_frequency != "daily":
        raise ValueError("Only rebalance_frequency: daily is supported in v1")
    if loaded.trigger_lookback_days <= 0:
        raise ValueError("trigger_lookback_days must be positive")
    if loaded.top_n <= 0:
        raise ValueError("top_n must be positive")
    if loaded.min_members_per_date <= 0 or loaded.min_members_per_date > loaded.top_n:
        raise ValueError("min_members_per_date must be between 1 and top_n")
    if loaded.exit_ween_days <= 0:
        raise ValueError("exit_ween_days must be positive")
    if loaded.buy_notional <= 0 or loaded.initial_cash <= 0:
        raise ValueError("initial_cash and buy_notional must be positive")
    return loaded
