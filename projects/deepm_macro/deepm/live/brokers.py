"""Execution broker interfaces for simulator and IBKR modes."""

from __future__ import annotations

import os
from datetime import date

from deepm.live.exceptions import BrokerUnavailable, SafetyError
from deepm.live.ledger import PortfolioStore
from deepm.live.types import Fill, Instrument, OrderIntent, Position


class ExecutionBroker:
    """Base broker interface."""

    def account_equity(self, latest_prices: dict[str, float]) -> float:
        raise NotImplementedError

    def current_positions(self, latest_prices: dict[str, float]) -> dict[str, Position]:
        raise NotImplementedError

    def execute_orders(self, orders: list[OrderIntent], run_date: date) -> list[Fill]:
        raise NotImplementedError


class SimBroker(ExecutionBroker):
    """Fake-money broker backed by the local SQLite ledger."""

    def __init__(
        self,
        store: PortfolioStore,
        instruments: dict[str, Instrument],
        mode: str = "sim",
        slippage_bps: float = 5.0,
    ):
        self.store = store
        self.instruments = instruments
        self.mode = mode
        self.slippage_bps = float(slippage_bps)

    @property
    def point_values(self) -> dict[str, float]:
        return {ticker: instrument.point_value for ticker, instrument in self.instruments.items()}

    def account_equity(self, latest_prices: dict[str, float]) -> float:
        return self.store.equity(latest_prices, self.point_values)

    def current_positions(self, latest_prices: dict[str, float]) -> dict[str, Position]:
        return self.store.positions(latest_prices, self.point_values)

    def execute_orders(self, orders: list[OrderIntent], run_date: date) -> list[Fill]:
        fills: list[Fill] = []
        for index, order in enumerate(orders, start=1):
            direction = 1 if order.side == "BUY" else -1
            price = order.market_price * (1.0 + direction * self.slippage_bps / 10000.0)
            commission = 0.0
            fill = Fill(
                fill_id=f"{run_date.isoformat()}:{self.mode}:{order.ticker}:{index}",
                run_date=run_date,
                mode=self.mode,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                price=round(price, 8),
                point_value=order.point_value,
                commission=commission,
                status="filled",
                order_id=f"SIM-{run_date.isoformat()}-{index}",
            )
            fills.append(fill)
        self.store.apply_fills(fills)
        return fills


class IBKRBroker(ExecutionBroker):
    """Safety-gated IBKR adapter shell.

    The integration is intentionally fail-closed unless the operator has TWS/Gateway,
    market data, account permissions, and the explicit environment gate enabled.
    """

    def __init__(self, mode: str, config: dict[str, object], instruments: dict[str, Instrument]):
        self.mode = mode
        self.config = config
        self.instruments = instruments

    def _ensure_enabled(self) -> None:
        if self.mode == "ibkr-live" and os.getenv("DEEPM_IBKR_ALLOW_LIVE") != "1":
            raise SafetyError("IBKR live mode also requires DEEPM_IBKR_ALLOW_LIVE=1")
        if os.getenv("DEEPM_IBKR_ALLOW_ORDERS") != "1":
            raise SafetyError(
                "IBKR order placement is disabled. Set DEEPM_IBKR_ALLOW_ORDERS=1 only after paper-mode validation."
            )
        try:
            import ibapi  # noqa: F401
        except ImportError as exc:
            raise BrokerUnavailable("ibapi is not installed in the active environment") from exc

    def account_equity(self, latest_prices: dict[str, float]) -> float:
        self._ensure_enabled()
        raise BrokerUnavailable(
            "IBKR account equity retrieval needs a connected TWS/Gateway session and account subscription wiring."
        )

    def current_positions(self, latest_prices: dict[str, float]) -> dict[str, Position]:
        self._ensure_enabled()
        raise BrokerUnavailable(
            "IBKR position reconciliation needs a connected TWS/Gateway session."
        )

    def execute_orders(self, orders: list[OrderIntent], run_date: date) -> list[Fill]:
        self._ensure_enabled()
        raise BrokerUnavailable(
            "IBKR order routing has not been enabled in this local run. Use simulator first, then implement/test TWS callbacks."
        )
