"""Risk sizing and order-intent generation."""

from __future__ import annotations

import math
from datetime import date
from typing import Iterable

import pandas as pd

from deepm.live.exceptions import SafetyError
from deepm.live.types import Instrument, OrderIntent, Position, RiskCheck, Signal, TargetPosition


def _daily_vol_from_row(row: pd.Series) -> float:
    daily_vol = float(row.get("daily_vol", 0.0) or 0.0)
    if daily_vol > 0:
        return daily_vol
    vs_factor = float(row.get("vs_factor", 0.0) or 0.0)
    if vs_factor > 0:
        return 1.0 / vs_factor
    return 0.0


def _market_price_from_row(row: pd.Series) -> float:
    for column in ("close", "srs", "market_price"):
        if column in row and pd.notna(row[column]):
            value = float(row[column])
            if value > 0:
                return value
    return 0.0


def build_targets(
    signals: Iterable[Signal],
    latest_features: pd.DataFrame,
    instruments: dict[str, Instrument],
    current_positions: dict[str, Position],
    account_equity: float,
    risk_config: dict[str, float | bool],
) -> tuple[list[TargetPosition], list[RiskCheck]]:
    """Convert model positions to integer contract targets."""
    signals = list(signals)
    active_assets = max(1, len(signals))
    target_daily_vol = float(risk_config["target_annual_vol"]) / math.sqrt(252.0)
    asset_risk_budget = account_equity * target_daily_vol / math.sqrt(active_assets)
    allow_fractional = bool(risk_config.get("allow_fractional_futures", False))

    targets: list[TargetPosition] = []
    checks: list[RiskCheck] = []
    for signal in signals:
        instrument = instruments.get(signal.ticker)
        if instrument is None:
            checks.append(
                RiskCheck(
                    name=f"{signal.ticker}.instrument_known",
                    passed=False,
                    value="missing",
                    limit="configured",
                    message="Ticker is missing from instrument map",
                )
            )
            continue
        if signal.ticker not in latest_features.index:
            checks.append(
                RiskCheck(
                    name=f"{signal.ticker}.feature_row",
                    passed=False,
                    value="missing",
                    limit="latest",
                    message="Ticker has no latest feature row",
                )
            )
            continue

        row = latest_features.loc[signal.ticker]
        daily_vol = _daily_vol_from_row(row)
        market_price = _market_price_from_row(row)
        if daily_vol <= 0 or market_price <= 0:
            checks.append(
                RiskCheck(
                    name=f"{signal.ticker}.sizing_inputs",
                    passed=False,
                    value={"daily_vol": daily_vol, "market_price": market_price},
                    limit="positive",
                    message="Daily vol and market price must be positive",
                )
            )
            continue

        contract_value = market_price * instrument.point_value
        target_notional = signal.model_position * asset_risk_budget / daily_vol
        raw_contracts = target_notional / contract_value
        target_quantity = int(round(raw_contracts))
        if allow_fractional:
            target_quantity = int(round(raw_contracts))
        target_quantity = max(-instrument.max_contracts, min(instrument.max_contracts, target_quantity))

        max_order_notional = float(risk_config.get("max_order_notional", 1.0))
        max_contracts_by_notional = int((account_equity * max_order_notional) // contract_value)
        if abs(target_quantity) > max_contracts_by_notional:
            capped_quantity = int(math.copysign(max_contracts_by_notional, target_quantity))
            checks.append(
                RiskCheck(
                    name=f"{signal.ticker}.integer_contract_notional_cap",
                    passed=True,
                    value=target_quantity,
                    limit=capped_quantity,
                    message="Target quantity capped to satisfy max_order_notional with integer contracts",
                )
            )
            target_quantity = capped_quantity

        current = current_positions.get(signal.ticker)
        targets.append(
            TargetPosition(
                ticker=signal.ticker,
                model_position=signal.model_position,
                current_quantity=current.quantity if current else 0,
                target_quantity=target_quantity,
                market_price=market_price,
                daily_vol=daily_vol,
                point_value=instrument.point_value,
                contract_value=contract_value,
                target_notional=target_notional,
                reviewed=instrument.reviewed,
            )
        )

    return targets, checks


def build_orders(
    targets: Iterable[TargetPosition],
    instruments: dict[str, Instrument],
    run_date: date,
    risk_config: dict[str, float | bool],
) -> list[OrderIntent]:
    """Create marketable limit order intents for target-current deltas."""
    orders: list[OrderIntent] = []
    ticks = float(risk_config.get("marketable_limit_ticks", 2))
    bps = float(risk_config.get("marketable_limit_bps", 5.0))
    for target in targets:
        delta = target.target_quantity - target.current_quantity
        if delta == 0:
            continue
        instrument = instruments[target.ticker]
        price_offset = max(ticks * instrument.tick_size, abs(target.market_price) * bps / 10000.0)
        side = "BUY" if delta > 0 else "SELL"
        limit_price = target.market_price + price_offset if side == "BUY" else target.market_price - price_offset
        orders.append(
            OrderIntent(
                run_date=run_date,
                ticker=target.ticker,
                side=side,
                quantity=abs(delta),
                current_quantity=target.current_quantity,
                target_quantity=target.target_quantity,
                limit_price=round(limit_price, 8),
                market_price=target.market_price,
                point_value=target.point_value,
                contract_value=target.contract_value,
                exchange=instrument.exchange,
                currency=instrument.currency,
            )
        )
    return orders


def risk_checks(
    targets: Iterable[TargetPosition],
    orders: Iterable[OrderIntent],
    account_equity: float,
    risk_config: dict[str, float | bool],
    mode: str,
) -> list[RiskCheck]:
    """Evaluate portfolio and order safety limits."""
    targets = list(targets)
    orders = list(orders)
    checks: list[RiskCheck] = []
    if account_equity <= 0:
        checks.append(
            RiskCheck(
                name="account_equity",
                passed=False,
                value=account_equity,
                limit="positive",
                message="Account equity must be positive",
            )
        )
        return checks

    gross = sum(abs(t.target_quantity * t.contract_value) for t in targets) / account_equity
    turnover = sum(abs(o.quantity * o.contract_value) for o in orders) / account_equity
    checks.append(
        RiskCheck(
            name="gross_leverage",
            passed=gross <= float(risk_config["max_gross_leverage"]),
            value=round(gross, 6),
            limit=float(risk_config["max_gross_leverage"]),
        )
    )
    checks.append(
        RiskCheck(
            name="daily_turnover",
            passed=turnover <= float(risk_config["max_daily_turnover"]),
            value=round(turnover, 6),
            limit=float(risk_config["max_daily_turnover"]),
        )
    )
    max_order = float(risk_config["max_order_notional"])
    for order in orders:
        ratio = abs(order.quantity * order.contract_value) / account_equity
        checks.append(
            RiskCheck(
                name=f"{order.ticker}.order_notional",
                passed=ratio <= max_order,
                value=round(ratio, 6),
                limit=max_order,
            )
        )
    if mode in {"ibkr-paper", "ibkr-live"}:
        unreviewed = [target.ticker for target in targets if not target.reviewed and target.target_quantity != 0]
        checks.append(
            RiskCheck(
                name="reviewed_contract_mapping",
                passed=not unreviewed,
                value=unreviewed,
                limit=[],
                message="Paper/live trading refuses unreviewed instrument mappings",
            )
        )
    return checks


def assert_risk_checks_pass(checks: Iterable[RiskCheck]) -> None:
    """Fail closed if any risk check did not pass."""
    failures = [check for check in checks if not check.passed]
    if failures:
        details = "; ".join(f"{check.name}={check.value} limit={check.limit}" for check in failures[:5])
        raise SafetyError(f"Risk checks failed: {details}")
