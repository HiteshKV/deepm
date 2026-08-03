"""SQLite simulator ledger with idempotent daily fills."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from deepm.live.types import Fill, Position


class PortfolioStore:
    """Persistent simulator ledger."""

    def __init__(self, path: str | Path, starting_cash: float):
        self.path = Path(path)
        self.starting_cash = float(starting_cash)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cash_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT DEFAULT CURRENT_TIMESTAMP,
                    amount REAL NOT NULL,
                    reason TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    run_date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    signed_quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    point_value REAL NOT NULL,
                    commission REAL NOT NULL,
                    status TEXT NOT NULL,
                    order_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (run_date, mode)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    run_date TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    PRIMARY KEY (run_date, mode)
                )
                """
            )
            if conn.execute("SELECT COUNT(*) FROM cash_ledger").fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO cash_ledger (amount, reason) VALUES (?, ?)",
                    (self.starting_cash, "initial_cash"),
                )

    def mark_run(self, run_date: date, mode: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_date, mode, status) VALUES (?, ?, ?)
                ON CONFLICT(run_date, mode) DO UPDATE SET status=excluded.status
                """,
                (run_date.isoformat(), mode, status),
            )

    def has_completed_run(self, run_date: date, mode: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM runs WHERE run_date=? AND mode=?",
                (run_date.isoformat(), mode),
            ).fetchone()
        return bool(row and row[0] == "completed")

    def latest_completed_run(self, mode: str) -> date | None:
        """Return the most recent successfully completed date for a mode."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(run_date)
                FROM runs
                WHERE mode=? AND status='completed'
                """,
                (mode,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return date.fromisoformat(str(row[0]))

    def cash(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(SUM(amount), 0.0) FROM cash_ledger").fetchone()
        return float(row[0])

    def positions(self, latest_prices: dict[str, float], point_values: dict[str, float]) -> dict[str, Position]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, SUM(signed_quantity) AS qty,
                       SUM(signed_quantity * price) / NULLIF(SUM(signed_quantity), 0) AS avg_price
                FROM fills
                WHERE status='filled'
                GROUP BY ticker
                HAVING qty != 0
                """
            ).fetchall()
        positions: dict[str, Position] = {}
        for ticker, quantity, avg_price in rows:
            price = float(latest_prices.get(ticker, avg_price or 0.0))
            point_value = float(point_values.get(ticker, 1.0))
            positions[ticker] = Position(
                ticker=ticker,
                quantity=int(quantity),
                avg_price=float(avg_price or price),
                point_value=point_value,
                market_price=price,
            )
        return positions

    def equity(self, latest_prices: dict[str, float], point_values: dict[str, float]) -> float:
        return self.cash() + sum(
            position.market_value for position in self.positions(latest_prices, point_values).values()
        )

    def previous_equity(self, run_date: date, mode: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT equity FROM equity_snapshots
                WHERE mode=? AND run_date < ?
                ORDER BY run_date DESC
                LIMIT 1
                """,
                (mode, run_date.isoformat()),
            ).fetchone()
        return None if row is None else float(row[0])

    def record_equity(self, run_date: date, mode: str, equity: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots (run_date, mode, equity, cash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_date, mode) DO UPDATE
                SET equity=excluded.equity, cash=excluded.cash
                """,
                (run_date.isoformat(), mode, float(equity), self.cash()),
            )

    def apply_fills(self, fills: list[Fill]) -> None:
        if not fills:
            return
        with self._connect() as conn:
            for fill in fills:
                signed_quantity = fill.quantity if fill.side == "BUY" else -fill.quantity
                cash_change = -(signed_quantity * fill.price * fill.point_value) - fill.commission
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO fills (
                        fill_id, run_date, mode, ticker, side, quantity, signed_quantity,
                        price, point_value, commission, status, order_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill.fill_id,
                        fill.run_date.isoformat(),
                        fill.mode,
                        fill.ticker,
                        fill.side,
                        fill.quantity,
                        signed_quantity,
                        fill.price,
                        fill.point_value,
                        fill.commission,
                        fill.status,
                        fill.order_id,
                    ),
                )
                if cursor.rowcount:
                    conn.execute(
                        "INSERT INTO cash_ledger (amount, reason) VALUES (?, ?)",
                        (cash_change, f"fill:{fill.fill_id}"),
                    )
