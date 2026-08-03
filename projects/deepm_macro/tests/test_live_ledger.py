import tempfile
import unittest
from datetime import date
from pathlib import Path

from deepm.live.ledger import PortfolioStore
from deepm.live.types import Fill


class LiveLedgerTests(unittest.TestCase):
    def test_fill_application_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioStore(Path(tmp) / "ledger.sqlite", starting_cash=1000.0)
            fill = Fill(
                fill_id="2026-06-25:sim:ES:1",
                run_date=date(2026, 6, 25),
                mode="sim",
                ticker="ES",
                side="BUY",
                quantity=2,
                price=10.0,
                point_value=1.0,
                commission=0.0,
                status="filled",
                order_id="SIM-1",
            )
            store.apply_fills([fill])
            store.apply_fills([fill])

            self.assertEqual(store.cash(), 980.0)
            positions = store.positions({"ES": 11.0}, {"ES": 1.0})
            self.assertEqual(positions["ES"].quantity, 2)
            self.assertEqual(positions["ES"].market_value, 22.0)

    def test_run_marker_prevents_duplicate_daily_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioStore(Path(tmp) / "ledger.sqlite", starting_cash=1000.0)
            run_date = date(2026, 6, 25)
            self.assertFalse(store.has_completed_run(run_date, "sim"))
            store.mark_run(run_date, "sim", "completed")
            self.assertTrue(store.has_completed_run(run_date, "sim"))

    def test_latest_completed_run_ignores_later_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioStore(Path(tmp) / "ledger.sqlite", starting_cash=1000.0)
            store.mark_run(date(2026, 6, 25), "sim", "completed")
            store.mark_run(date(2026, 6, 26), "sim", "failed")

            self.assertEqual(store.latest_completed_run("sim"), date(2026, 6, 25))
            self.assertIsNone(store.latest_completed_run("ibkr-paper"))

    def test_equity_snapshots_return_previous_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PortfolioStore(Path(tmp) / "ledger.sqlite", starting_cash=1000.0)
            store.record_equity(date(2026, 6, 24), "sim", 1005.0)
            store.record_equity(date(2026, 6, 25), "sim", 1010.0)
            self.assertEqual(store.previous_equity(date(2026, 6, 25), "sim"), 1005.0)


if __name__ == "__main__":
    unittest.main()
