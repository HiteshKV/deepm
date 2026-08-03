import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from stock_top10_strategy.config import load_config
from stock_top10_strategy.data import DataValidationError, load_point_in_time_data
from stock_top10_strategy.engine import run_backtest


def _write_dataset(path: Path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=8)
    rows = []
    caps = {
        "S01": 1200,
        "S02": 1100,
        "S03": 1000,
        "S04": 900,
        "S05": 800,
        "S06": 700,
        "S07": 600,
        "S08": 500,
        "S09": 400,
        "S10": 300,
        "S11": 200,
    }
    prices = {
        "S01": [100, 101, 102, 103, 104, 105, 106, 107],
        "S02": [100, 99, 98, 97, 96, 95, 96, 97],
        "S03": [50, 50, 50, 50, 50, 50, 50, 50],
        "S04": [40, 40, 40, 40, 40, 40, 40, 40],
        "S05": [30, 30, 30, 30, 30, 30, 30, 30],
        "S06": [20, 20, 20, 20, 20, 20, 20, 20],
        "S07": [10, 10, 10, 10, 10, 10, 10, 10],
        "S08": [9, 9, 9, 9, 9, 9, 9, 9],
        "S09": [8, 8, 8, 8, 8, 8, 8, 8],
        "S10": [7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7],
        "S11": [6, 6, 6, 6, 6, 6, 6, 6],
    }
    for i, day in enumerate(dates):
        for sid, base_cap in caps.items():
            cap = base_cap
            if sid == "S11" and i >= 5:
                cap = 350
            price = prices[sid][i]
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "security_id": sid,
                    "ticker": sid,
                    "company_name": f"Company {sid}",
                    "country": "US",
                    "currency": "USD",
                    "market_cap_usd": cap * 1_000_000,
                    "open": price,
                    "close": price,
                    "adj_close": price,
                    "fx_to_usd": 1.0,
                    "split_factor": 1.0,
                    "dividend": 0.0,
                    "is_active": True,
                    "delist_date": "",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_config(tmp: Path, data_path: Path, **overrides) -> Path:
    config = {
        "run_name": "unit_top10",
        "data": {
            "path": str(data_path),
            "source_label": "unit-test",
            "source_kind": "point_in_time",
            "start_date": None,
            "end_date": None,
        },
        "strategy": {
            "initial_cash": 100000.0,
            "buy_notional": 10000.0,
            "trigger_return": 0.03,
            "trigger_lookback_days": 5,
            "top_n": 10,
            "rebalance_frequency": "daily",
            "exit_ween_days": 10,
        },
        "execution": {
            "commission_bps": 1.0,
            "min_commission": 1.0,
            "half_spread_bps": 2.0,
            "slippage_bps": 3.0,
        },
        "risk": {
            "max_position_pct_nav": 0.5,
            "max_single_day_turnover_pct_nav": 0.8,
            "allow_margin": False,
            "allow_fractional_shares": False,
            "risk_free_rate": 0.0,
        },
        "reporting": {"output_dir": str(tmp / "runs")},
    }
    for section, values in overrides.items():
        config[section].update(values)
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class StockTop10StrategyTests(unittest.TestCase):
    def test_loader_rejects_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            data_path = tmp / "data.csv"
            _write_dataset(data_path)
            frame = pd.read_csv(data_path)
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            frame.to_csv(data_path, index=False)
            config = load_config(_write_config(tmp, data_path))
            with self.assertRaisesRegex(DataValidationError, "Duplicate"):
                load_point_in_time_data(config)

    def test_backtest_generates_move_buys_and_left_top10_exit(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            data_path = tmp / "data.csv"
            _write_dataset(data_path)
            config = load_config(_write_config(tmp, data_path))
            run_dir = run_backtest(config, run_id="unit_run")
            orders = pd.read_csv(run_dir / "orders.csv")
            fills = pd.read_csv(run_dir / "fills.csv")
            membership = pd.read_csv(run_dir / "membership.csv")

            self.assertIn("up_move_buy", set(orders["reason"]))
            self.assertIn("down_move_buy", set(orders["reason"]))
            self.assertIn("left_top10", set(orders["reason"]))
            self.assertTrue((fills["commission_usd"] >= 1.0).all())
            self.assertTrue((fills["execution_cost_usd"] >= 0.0).all())
            latest_members = membership[membership["date"] == membership["date"].max()]
            self.assertIn("S11", set(latest_members["security_id"]))

    def test_cash_limits_prevent_margin(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            data_path = tmp / "data.csv"
            _write_dataset(data_path)
            config_path = _write_config(
                tmp,
                data_path,
                strategy={"initial_cash": 1000.0, "buy_notional": 50000.0},
                risk={"allow_margin": False},
            )
            run_dir = run_backtest(load_config(config_path), run_id="cash_limited")
            nav = pd.read_csv(run_dir / "nav.csv")
            self.assertGreaterEqual(nav["cash"].min(), -1e-6)

    def test_default_sample_config_runs(self):
        config = load_config("stock_top10_strategy/configs/default.yaml")
        data = load_point_in_time_data(config)
        self.assertGreaterEqual(data["security_id"].nunique(), 10)


if __name__ == "__main__":
    unittest.main()
