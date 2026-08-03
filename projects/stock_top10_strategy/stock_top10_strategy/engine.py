"""Backtest engine for the top-10 buy-the-move strategy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_top10_strategy.config import StrategyConfig
from stock_top10_strategy.data import load_point_in_time_data
from stock_top10_strategy.reporting import build_report


@dataclass
class Position:
    security_id: str
    ticker: str
    company_name: str
    quantity: float
    avg_price_usd: float


@dataclass
class Order:
    signal_date: pd.Timestamp
    fill_date: pd.Timestamp
    security_id: str
    ticker: str
    company_name: str
    side: str
    quantity: float
    reason: str
    trigger_return: float | None


def run_backtest(config: StrategyConfig, run_id: str | None = None) -> Path:
    """Run a complete backtest and return the run directory."""
    data = load_point_in_time_data(config)
    run_id = run_id or f"{config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results = BacktestEngine(config, data, run_dir).run()
    write_outputs(run_dir, config, results)
    build_report(run_dir)
    return run_dir


class BacktestEngine:
    """Stateful point-in-time portfolio simulation."""

    def __init__(self, config: StrategyConfig, data: pd.DataFrame, run_dir: Path) -> None:
        self.config = config
        self.data = data
        self.run_dir = run_dir
        self.by_date = {
            date: frame.set_index("security_id", drop=False)
            for date, frame in data.groupby("date", sort=True)
        }
        self.dates = sorted(self.by_date)
        self.cash = config.initial_cash
        self.positions: dict[str, Position] = {}
        self.pending_orders: dict[pd.Timestamp, list[Order]] = {}
        self.exit_plans: dict[str, int] = {}
        self.orders: list[dict[str, object]] = []
        self.fills: list[dict[str, object]] = []
        self.nav_rows: list[dict[str, object]] = []
        self.holding_rows: list[dict[str, object]] = []
        self.membership_rows: list[dict[str, object]] = []
        self.warnings: list[str] = []

    def run(self) -> dict[str, object]:
        for i, current_date in enumerate(self.dates):
            today = self.by_date[current_date]
            self._apply_corporate_actions(current_date, today)
            self._credit_dividends(current_date, today)
            self._execute_pending_orders(current_date, today)
            nav = self._nav(today)
            self._record_nav(current_date, nav, today)

            next_date = self.dates[i + 1] if i + 1 < len(self.dates) else None
            if next_date is not None:
                orders = self._generate_orders(current_date, next_date, nav)
                self.pending_orders.setdefault(next_date, []).extend(orders)
                self.orders.extend(order.__dict__ for order in orders)

        nav = pd.DataFrame(self.nav_rows)
        fills = pd.DataFrame(self.fills)
        orders = pd.DataFrame(self.orders)
        holdings = pd.DataFrame(self.holding_rows)
        membership = pd.DataFrame(self.membership_rows)
        metrics = compute_metrics(nav, fills, self.config)
        return {
            "nav": nav,
            "orders": orders,
            "fills": fills,
            "holdings": holdings,
            "membership": membership,
            "metrics": metrics,
            "warnings": self.warnings,
        }

    def _apply_corporate_actions(self, current_date: pd.Timestamp, today: pd.DataFrame) -> None:
        for security_id, position in list(self.positions.items()):
            if security_id not in today.index:
                continue
            split_factor = float(today.loc[security_id, "split_factor"])
            if split_factor != 1.0:
                position.quantity *= split_factor
                position.avg_price_usd /= split_factor
                self.warnings.append(
                    f"{current_date.date()} split adjustment {security_id}: factor={split_factor}"
                )

    def _credit_dividends(self, current_date: pd.Timestamp, today: pd.DataFrame) -> None:
        for security_id, position in self.positions.items():
            if security_id not in today.index:
                continue
            row = today.loc[security_id]
            dividend_usd = float(row["dividend"]) * float(row["fx_to_usd"])
            if dividend_usd > 0:
                self.cash += position.quantity * dividend_usd

    def _execute_pending_orders(self, current_date: pd.Timestamp, today: pd.DataFrame) -> None:
        orders = self.pending_orders.pop(current_date, [])
        if not orders:
            return
        nav_start = self._nav(today)
        turnover_limit = nav_start * self.config.max_single_day_turnover_pct_nav
        used_turnover = 0.0
        for order in orders:
            if order.security_id not in today.index:
                self.warnings.append(
                    f"{current_date.date()} skipped order for missing security {order.security_id}"
                )
                continue
            row = today.loc[order.security_id]
            reference_price = float(row["price_open_usd"])
            used_fallback_close = bool(pd.isna(row.get("open", np.nan)))
            if used_fallback_close:
                reference_price = float(row["price_close_usd"])
            fill = self._fill_order(order, row, reference_price, used_turnover, turnover_limit)
            used_turnover += float(fill.get("notional_usd", 0.0))
            if fill["status"] == "filled":
                self._apply_fill(order, fill)
            self.fills.append(fill)

    def _fill_order(
        self,
        order: Order,
        row: pd.Series,
        reference_price: float,
        used_turnover: float,
        turnover_limit: float,
    ) -> dict[str, object]:
        side_mult = 1 if order.side == "BUY" else -1
        spread_bps = self.config.half_spread_bps
        slippage_bps = self.config.slippage_bps
        fill_price = reference_price * (1 + side_mult * (spread_bps + slippage_bps) / 10000)
        requested_qty = float(order.quantity)

        current_turnover_left = max(0.0, turnover_limit - used_turnover)
        max_qty_turnover = current_turnover_left / fill_price if turnover_limit > 0 else requested_qty
        quantity = min(requested_qty, max_qty_turnover)

        if order.side == "SELL":
            held = self.positions.get(order.security_id)
            quantity = min(quantity, held.quantity if held else 0.0)
        if order.side == "BUY" and not self.config.allow_margin:
            max_cash_qty = self.cash / self._cash_cost_per_share(fill_price)
            quantity = min(quantity, max_cash_qty)
        if not self.config.allow_fractional_shares:
            quantity = np.floor(quantity)

        if quantity <= 0:
            return self._fill_dict(order, row, reference_price, fill_price, 0.0, 0.0, 0.0, 0.0, "skipped")

        notional = quantity * fill_price
        commission = max(self.config.min_commission, notional * self.config.commission_bps / 10000)
        spread_cost = quantity * reference_price * spread_bps / 10000
        slippage_cost = quantity * reference_price * slippage_bps / 10000
        return self._fill_dict(
            order,
            row,
            reference_price,
            fill_price,
            quantity,
            notional,
            commission,
            spread_cost + slippage_cost,
            "filled",
        )

    def _fill_dict(
        self,
        order: Order,
        row: pd.Series,
        reference_price: float,
        fill_price: float,
        quantity: float,
        notional: float,
        commission: float,
        execution_cost: float,
        status: str,
    ) -> dict[str, object]:
        return {
            "signal_date": order.signal_date.date().isoformat(),
            "fill_date": order.fill_date.date().isoformat(),
            "security_id": order.security_id,
            "ticker": order.ticker,
            "company_name": order.company_name,
            "side": order.side,
            "quantity": quantity,
            "reference_price_usd": reference_price,
            "fill_price_usd": fill_price,
            "notional_usd": notional,
            "commission_usd": commission,
            "execution_cost_usd": execution_cost,
            "reason": order.reason,
            "trigger_return": order.trigger_return,
            "status": status,
            "currency": row["currency"],
        }

    def _apply_fill(self, order: Order, fill: dict[str, object]) -> None:
        quantity = float(fill["quantity"])
        price = float(fill["fill_price_usd"])
        commission = float(fill["commission_usd"])
        if order.side == "BUY":
            total_cost = quantity * price + commission
            self.cash -= total_cost
            current = self.positions.get(order.security_id)
            if current is None:
                self.positions[order.security_id] = Position(
                    order.security_id,
                    order.ticker,
                    order.company_name,
                    quantity,
                    price,
                )
            else:
                new_qty = current.quantity + quantity
                current.avg_price_usd = ((current.quantity * current.avg_price_usd) + (quantity * price)) / new_qty
                current.quantity = new_qty
        else:
            current = self.positions.get(order.security_id)
            if current is None:
                return
            proceeds = quantity * price - commission
            self.cash += proceeds
            current.quantity -= quantity
            if current.quantity <= 1e-9:
                self.positions.pop(order.security_id, None)
                self.exit_plans.pop(order.security_id, None)

    def _cash_cost_per_share(self, fill_price: float) -> float:
        return fill_price * (1 + self.config.commission_bps / 10000)

    def _generate_orders(
        self,
        signal_date: pd.Timestamp,
        fill_date: pd.Timestamp,
        nav: float,
    ) -> list[Order]:
        top_ids = self._top_members(signal_date)
        today = self.by_date[signal_date]
        orders: list[Order] = []
        for rank, security_id in enumerate(top_ids, start=1):
            row = today.loc[security_id]
            self.membership_rows.append(
                {
                    "date": signal_date.date().isoformat(),
                    "rank": rank,
                    "security_id": security_id,
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "market_cap_usd": row["market_cap_usd"],
                }
            )

        top_set = set(top_ids)
        for security_id in list(self.exit_plans):
            if security_id in top_set:
                self.exit_plans.pop(security_id, None)

        for security_id, position in list(self.positions.items()):
            if security_id in top_set or position.quantity <= 0:
                continue
            remaining_days = self.exit_plans.setdefault(security_id, self.config.exit_ween_days)
            sell_qty = position.quantity / max(1, remaining_days)
            if not self.config.allow_fractional_shares:
                sell_qty = np.ceil(sell_qty)
            orders.append(self._make_order(signal_date, fill_date, security_id, "SELL", sell_qty, "left_top10", None))
            self.exit_plans[security_id] = max(1, remaining_days - 1)

        for security_id in top_ids:
            trigger = self._rolling_return(security_id, signal_date)
            if trigger is None:
                continue
            if trigger >= self.config.trigger_return or trigger <= -self.config.trigger_return:
                qty = self._buy_quantity(security_id, signal_date, nav)
                if qty > 0:
                    reason = "up_move_buy" if trigger >= self.config.trigger_return else "down_move_buy"
                    orders.append(self._make_order(signal_date, fill_date, security_id, "BUY", qty, reason, trigger))
        return orders

    def _make_order(
        self,
        signal_date: pd.Timestamp,
        fill_date: pd.Timestamp,
        security_id: str,
        side: str,
        quantity: float,
        reason: str,
        trigger_return: float | None,
    ) -> Order:
        row = self.by_date[signal_date].loc[security_id]
        return Order(
            signal_date=signal_date,
            fill_date=fill_date,
            security_id=security_id,
            ticker=row["ticker"],
            company_name=row["company_name"],
            side=side,
            quantity=float(quantity),
            reason=reason,
            trigger_return=trigger_return,
        )

    def _top_members(self, signal_date: pd.Timestamp) -> list[str]:
        index = self.dates.index(signal_date)
        if index == 0:
            source_date = signal_date
        else:
            source_date = self.dates[index - 1]
        source = self.by_date[source_date]
        eligible = source[source["is_active"]].sort_values("market_cap_usd", ascending=False)
        return eligible.head(self.config.top_n)["security_id"].tolist()

    def _rolling_return(self, security_id: str, signal_date: pd.Timestamp) -> float | None:
        index = self.dates.index(signal_date)
        lookback_index = index - self.config.trigger_lookback_days
        if lookback_index < 0:
            return None
        start_date = self.dates[lookback_index]
        if security_id not in self.by_date[start_date].index or security_id not in self.by_date[signal_date].index:
            return None
        start = float(self.by_date[start_date].loc[security_id, "adj_close_usd"])
        end = float(self.by_date[signal_date].loc[security_id, "adj_close_usd"])
        return end / start - 1.0

    def _buy_quantity(self, security_id: str, signal_date: pd.Timestamp, nav: float) -> float:
        row = self.by_date[signal_date].loc[security_id]
        price = float(row["price_close_usd"])
        current = self.positions.get(security_id)
        current_value = (current.quantity * price) if current else 0.0
        max_value = nav * self.config.max_position_pct_nav
        available_value = max(0.0, max_value - current_value)
        notional = min(self.config.buy_notional, available_value, self.cash if not self.config.allow_margin else self.config.buy_notional)
        qty = notional / price
        if not self.config.allow_fractional_shares:
            qty = np.floor(qty)
        return float(qty)

    def _nav(self, today: pd.DataFrame) -> float:
        return self.cash + self._market_value(today)

    def _market_value(self, today: pd.DataFrame) -> float:
        value = 0.0
        for security_id, position in self.positions.items():
            if security_id in today.index:
                value += position.quantity * float(today.loc[security_id, "price_close_usd"])
        return value

    def _record_nav(self, current_date: pd.Timestamp, nav: float, today: pd.DataFrame) -> None:
        market_value = self._market_value(today)
        self.nav_rows.append(
            {
                "date": current_date.date().isoformat(),
                "nav": nav,
                "cash": self.cash,
                "market_value": market_value,
                "gross_exposure": abs(market_value),
                "cash_pct": self.cash / nav if nav else np.nan,
                "positions": len(self.positions),
            }
        )
        for security_id, position in self.positions.items():
            if security_id not in today.index:
                continue
            price = float(today.loc[security_id, "price_close_usd"])
            self.holding_rows.append(
                {
                    "date": current_date.date().isoformat(),
                    "security_id": security_id,
                    "ticker": position.ticker,
                    "company_name": position.company_name,
                    "quantity": position.quantity,
                    "avg_price_usd": position.avg_price_usd,
                    "close_price_usd": price,
                    "market_value_usd": position.quantity * price,
                    "unrealized_pnl_usd": position.quantity * (price - position.avg_price_usd),
                }
            )


def compute_metrics(nav: pd.DataFrame, fills: pd.DataFrame, config: StrategyConfig) -> dict[str, float | int | str]:
    if nav.empty:
        return {}
    curve = nav.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve["return"] = curve["nav"].pct_change().fillna(0.0)
    years = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days / 365.25, 1 / 252)
    total_return = curve["nav"].iloc[-1] / curve["nav"].iloc[0] - 1.0
    cagr = (curve["nav"].iloc[-1] / curve["nav"].iloc[0]) ** (1 / years) - 1.0
    vol = curve["return"].std(ddof=0) * np.sqrt(252)
    excess_daily = config.risk_free_rate / 252
    sharpe = ((curve["return"].mean() - excess_daily) / curve["return"].std(ddof=0) * np.sqrt(252)) if curve["return"].std(ddof=0) else np.nan
    running_max = curve["nav"].cummax()
    drawdown = curve["nav"] / running_max - 1.0
    total_costs = 0.0
    turnover = 0.0
    filled_trades = 0
    if not fills.empty and "status" in fills:
        filled = fills[fills["status"] == "filled"]
        total_costs = float(filled["commission_usd"].sum() + filled["execution_cost_usd"].sum())
        turnover = float(filled["notional_usd"].sum())
        filled_trades = int(len(filled))
    return {
        "start_date": curve["date"].iloc[0].date().isoformat(),
        "end_date": curve["date"].iloc[-1].date().isoformat(),
        "initial_nav": float(curve["nav"].iloc[0]),
        "ending_nav": float(curve["nav"].iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_vol": float(vol),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown": float(drawdown.min()),
        "filled_trades": filled_trades,
        "total_turnover_usd": turnover,
        "total_costs_usd": total_costs,
        "cash_drag_avg": float(curve["cash_pct"].mean()),
        "data_source_label": config.data_source_label,
        "data_source_kind": config.data_source_kind,
    }


def write_outputs(run_dir: Path, config: StrategyConfig, results: dict[str, object]) -> None:
    """Write all run artifacts."""
    (run_dir / "config.json").write_text(json.dumps(config.__dict__, default=str, indent=2), encoding="utf-8")
    for name in ("nav", "orders", "fills", "holdings", "membership"):
        frame = results[name]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(run_dir / f"{name}.csv", index=False)
    metrics = results["metrics"]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    warnings = results.get("warnings", [])
    (run_dir / "warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
