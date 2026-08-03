"""Small data contracts shared by the live trading modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Instrument:
    ticker: str
    bloomberg_ticker: str
    yahoo_symbol: str
    description: str
    ibkr_symbol: str
    sec_type: str
    exchange: str
    currency: str
    point_value: float
    tick_size: float
    contract_type: str
    max_contracts: int
    reviewed: bool


@dataclass(frozen=True)
class ModelWindow:
    start_year: int
    end_year: Optional[int]
    path: Path
    run_names: list[str]


@dataclass(frozen=True)
class Signal:
    run_date: date
    ticker: str
    model_position: float


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: int
    avg_price: float
    point_value: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price * self.point_value


@dataclass(frozen=True)
class TargetPosition:
    ticker: str
    model_position: float
    current_quantity: int
    target_quantity: int
    market_price: float
    daily_vol: float
    point_value: float
    contract_value: float
    target_notional: float
    reviewed: bool


@dataclass(frozen=True)
class OrderIntent:
    run_date: date
    ticker: str
    side: str
    quantity: int
    current_quantity: int
    target_quantity: int
    limit_price: float
    market_price: float
    point_value: float
    contract_value: float
    exchange: str
    currency: str
    con_id: Optional[int] = None
    expiry: Optional[str] = None
    reason: str = "rebalance"


@dataclass(frozen=True)
class Fill:
    fill_id: str
    run_date: date
    mode: str
    ticker: str
    side: str
    quantity: int
    price: float
    point_value: float
    commission: float
    status: str
    order_id: Optional[str] = None


@dataclass(frozen=True)
class RiskCheck:
    name: str
    passed: bool
    value: Any
    limit: Any
    message: str = ""


def dataclass_to_dict(value: Any) -> Dict[str, Any]:
    """Convert a dataclass to a JSON-friendly dictionary."""
    data = asdict(value)
    for key, item in list(data.items()):
        if isinstance(item, date):
            data[key] = item.isoformat()
        elif isinstance(item, Path):
            data[key] = str(item)
    return data
