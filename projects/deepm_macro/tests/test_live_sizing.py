import unittest
from datetime import date

import pandas as pd

from deepm.live.sizing import build_orders, build_targets, risk_checks
from deepm.live.types import Instrument, Position, RiskCheck, Signal, TargetPosition


def instrument(ticker="ES", reviewed=True):
    return Instrument(
        ticker=ticker,
        bloomberg_ticker=f"{ticker}1 Index",
        yahoo_symbol=f"{ticker}=F",
        description=ticker,
        ibkr_symbol=ticker,
        sec_type="FUT",
        exchange="CME",
        currency="USD",
        point_value=1.0,
        tick_size=0.25,
        contract_type="FUT",
        max_contracts=5,
        reviewed=reviewed,
    )


class LiveSizingTests(unittest.TestCase):
    def test_position_sizing_caps_contracts_and_builds_order(self):
        instruments = {"ES": instrument()}
        latest = pd.DataFrame(
            [{"ticker": "ES", "close": 100.0, "daily_vol": 0.01}]
        ).set_index("ticker", drop=False)
        risk = {
            "target_annual_vol": 0.10,
            "max_gross_leverage": 1.0,
            "max_daily_turnover": 0.20,
            "max_order_notional": 0.10,
            "allow_fractional_futures": False,
            "marketable_limit_ticks": 2,
            "marketable_limit_bps": 5.0,
        }
        targets, checks = build_targets(
            [Signal(date(2026, 6, 25), "ES", 1.0)],
            latest,
            instruments,
            {},
            100_000.0,
            risk,
        )
        self.assertEqual(checks, [])
        self.assertEqual(targets[0].target_quantity, 5)

        orders = build_orders(targets, instruments, date(2026, 6, 25), risk)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "BUY")
        self.assertGreater(orders[0].limit_price, orders[0].market_price)

    def test_zero_vol_creates_failed_input_check(self):
        instruments = {"ES": instrument()}
        latest = pd.DataFrame(
            [{"ticker": "ES", "close": 100.0, "daily_vol": 0.0}]
        ).set_index("ticker", drop=False)
        targets, checks = build_targets(
            [Signal(date(2026, 6, 25), "ES", 1.0)],
            latest,
            instruments,
            {},
            100_000.0,
            {"target_annual_vol": 0.10},
        )
        self.assertEqual(targets, [])
        self.assertFalse(checks[0].passed)

    def test_integer_contract_above_order_cap_is_skipped(self):
        instruments = {"CB": instrument("CB")}
        instruments["CB"] = Instrument(
            **{
                **instruments["CB"].__dict__,
                "point_value": 1000.0,
                "max_contracts": 5,
            }
        )
        latest = pd.DataFrame(
            [{"ticker": "CB", "close": 19.35, "daily_vol": 0.002}]
        ).set_index("ticker", drop=False)
        targets, checks = build_targets(
            [Signal(date(2026, 6, 25), "CB", 1.0)],
            latest,
            instruments,
            {},
            100_000.0,
            {
                "target_annual_vol": 0.10,
                "max_order_notional": 0.10,
                "allow_fractional_futures": False,
            },
        )
        self.assertEqual(targets[0].target_quantity, 0)
        self.assertTrue(any(check.name == "CB.integer_contract_notional_cap" for check in checks))

    def test_reversal_order_delta_is_capped_to_order_notional(self):
        instruments = {"UZ": instrument("UZ")}
        latest = pd.DataFrame(
            [{"ticker": "UZ", "close": 80.18, "daily_vol": 0.002}]
        ).set_index("ticker", drop=False)
        risk = {
            "target_annual_vol": 0.10,
            "max_gross_leverage": 1.0,
            "max_daily_turnover": 0.50,
            "max_order_notional": 0.10,
            "allow_fractional_futures": False,
            "marketable_limit_ticks": 2,
            "marketable_limit_bps": 5.0,
        }
        current_positions = {
            "UZ": Position(
                ticker="UZ",
                quantity=-1,
                avg_price=80.0,
                point_value=1.0,
                market_price=80.18,
            )
        }
        targets, _checks = build_targets(
            [Signal(date(2026, 7, 1), "UZ", 1.0)],
            latest,
            instruments,
            current_positions,
            1_000.0,
            risk,
        )
        self.assertEqual(targets[0].current_quantity, -1)
        self.assertEqual(targets[0].target_quantity, 1)

        orders = build_orders(targets, instruments, date(2026, 7, 1), risk, 1_000.0)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "BUY")
        self.assertEqual(orders[0].quantity, 1)
        self.assertEqual(orders[0].target_quantity, 0)
        self.assertEqual(orders[0].reason, "rebalance_order_delta_capped_to_max_notional")
        checks = risk_checks(targets, orders, 1_000.0, risk, mode="sim")
        self.assertTrue(all(check.passed for check in checks))

    def test_paper_mode_fails_unreviewed_contract_mapping(self):
        targets = [
            TargetPosition(
                ticker="LX",
                model_position=1.0,
                current_quantity=0,
                target_quantity=1,
                market_price=100.0,
                daily_vol=0.01,
                point_value=10.0,
                contract_value=1000.0,
                target_notional=1000.0,
                reviewed=False,
            )
        ]
        checks = risk_checks(
            targets,
            [],
            100_000.0,
            {
                "max_gross_leverage": 1.0,
                "max_daily_turnover": 0.20,
                "max_order_notional": 0.10,
            },
            mode="ibkr-paper",
        )
        self.assertTrue(any(isinstance(check, RiskCheck) for check in checks))
        self.assertFalse([check for check in checks if check.name == "reviewed_contract_mapping"][0].passed)


if __name__ == "__main__":
    unittest.main()
