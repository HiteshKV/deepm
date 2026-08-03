from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import unittest

from equity_intraday_deepm.backtest import run_backtest
from equity_intraday_deepm.config import load_data_config
from equity_intraday_deepm.data import build_universe, validate_source
from equity_intraday_deepm.features import build_features
from equity_intraday_deepm.finalize import finalize_run
from equity_intraday_deepm.model import sharpe_score
from equity_intraday_deepm.training import train_from_config
from equity_intraday_deepm.vendor_import import build_canonical_from_vendor_exports, import_vendor_exports


class EquityIntradayDeePMTests(unittest.TestCase):
    def test_causal_top10_membership_and_inactive_price_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = _write_source(root)
            cfg = load_data_config(cfg_path)
            summary = validate_source(cfg)
            self.assertEqual(summary["tradable_common_equities"], 3)

            universe_path, audit_path = build_universe(cfg, "2000-12-20", "2001-01-03", active_top_n=2)
            universe = pd.read_parquet(universe_path)
            first = universe[universe["timestamp"] == universe["timestamp"].min()].sort_values("rank")
            self.assertEqual(first["security_id"].tolist(), ["AAA", "BBB"])
            later = universe[universe["timestamp"] == pd.Timestamp("2001-01-02 10:00:00")].sort_values("rank")
            self.assertEqual(later["security_id"].tolist(), ["CCC", "AAA"])
            audit = json.loads(Path(audit_path).read_text())
            self.assertEqual(audit["active_top_n"], 2)

            features_path, _ = build_features(cfg)
            features = pd.read_parquet(features_path)
            inactive_bbb = features[
                (features["timestamp"] == pd.Timestamp("2001-01-02 10:00:00"))
                & (features["security_id"] == "BBB")
            ]
            self.assertFalse(bool(inactive_bbb["is_active"].iloc[0]))
            self.assertAlmostEqual(float(features["transaction_cost_bps"].iloc[0]), 3.5)

    def test_bars_per_year_changes_sharpe_scaling(self):
        values = np.array([0.001, -0.0005, 0.0015, 0.0002], dtype=np.float32)
        import torch

        daily = sharpe_score(torch.tensor(values), 252.0).item()
        hourly = sharpe_score(torch.tensor(values), 1638.0).item()
        self.assertGreater(hourly, daily)

    def test_smoke_train_finalize_backtest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_cfg = load_data_config(_write_source(root))
            build_universe(data_cfg, "2000-12-20", "2001-01-04", active_top_n=2)
            build_features(data_cfg)
            train_cfg = _write_train(root, data_cfg.features_output_path)
            train_from_config(str(train_cfg), "EQH_DEEPM_XATT", "cpu", filter_start_years=[2001])
            finalized = finalize_run(str(train_cfg), top_n=1, allow_partial=True)
            self.assertTrue(finalized)
            bt_cfg = _write_backtest(root, train_cfg, data_cfg.features_output_path)
            out_dir = run_backtest(str(bt_cfg), diagnostics=True)
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "nav_drawdown.png").exists())

    def test_vendor_import_writes_canonical_files_and_audits_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_cfg = load_data_config(_write_source(root))
            import_cfg = root / "import.yaml"
            import_cfg.write_text(
                f"""
start: "2000-12-20"
end: "2001-01-04"
active_top_n: 2
strict_top_n: true
inputs:
  security_master: {source_cfg.security_master_path}
  market_caps: {source_cfg.market_caps_path}
  hourly_bars: {source_cfg.hourly_bars_path}
outputs:
  security_master: {root / 'vendor' / 'security_master.parquet'}
  market_caps: {root / 'vendor' / 'market_caps.parquet'}
  hourly_bars: {root / 'vendor' / 'hourly_bars.parquet'}
column_maps:
  security_master: {{}}
  market_caps: {{}}
  hourly_bars: {{}}
""",
                encoding="utf-8",
            )
            audit = import_vendor_exports(import_cfg)
            self.assertTrue(Path(audit["outputs"]["security_master"]).exists())
            self.assertTrue(Path(audit["outputs"]["market_caps"]).exists())
            self.assertTrue(Path(audit["outputs"]["hourly_bars"]).exists())
            self.assertEqual(audit["min_active_members"], 2)

    def test_build_canonical_computes_market_caps_and_consolidates_companies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = pd.DataFrame(
                [
                    ["MEGA", "AAA", "AAA", "AAA", "Alpha A", "US", "Tech", "USD", "NASDAQ", "1001", True, True, True, "1990-01-01", ""],
                    ["MEGA", "AAB", "AAA", "AAB", "Alpha B", "US", "Tech", "USD", "NASDAQ", "1004", True, True, False, "1990-01-01", ""],
                    ["BETA", "BBB", "BBB", "BBB", "Beta", "US", "Tech", "USD", "NYSE", "1002", True, True, True, "1990-01-01", ""],
                    ["GAMMA", "CCC", "CCC", "CCC", "Gamma", "JP", "Health", "JPY", "TSE", "1003", True, True, True, "1990-01-01", ""],
                ],
                columns=[
                    "company_id",
                    "security_id",
                    "primary_security_id",
                    "ticker",
                    "company_name",
                    "country",
                    "sector",
                    "currency",
                    "exchange",
                    "ibkr_con_id",
                    "is_common_equity",
                    "is_ibkr_tradable",
                    "is_primary_listing",
                    "listing_start",
                    "listing_end",
                ],
            )
            prices = pd.DataFrame(
                [
                    ["1999-12-31", "AAA", 50.0, "USD", 1.0],
                    ["1999-12-31", "AAB", 30.0, "USD", 1.0],
                    ["1999-12-31", "BBB", 70.0, "USD", 1.0],
                    ["1999-12-31", "CCC", 1000.0, "JPY", np.nan],
                    ["2000-12-31", "AAA", 60.0, "USD", 1.0],
                    ["2000-12-31", "AAB", 40.0, "USD", 1.0],
                    ["2000-12-31", "BBB", 65.0, "USD", 1.0],
                    ["2000-12-31", "CCC", 1200.0, "JPY", np.nan],
                ],
                columns=["date", "security_id", "close", "currency", "fx_to_usd"],
            )
            shares = pd.DataFrame(
                [
                    ["1999-12-01", "AAA", 2.0],
                    ["1999-12-01", "AAB", 2.0],
                    ["1999-12-01", "BBB", 3.0],
                    ["1999-12-01", "CCC", 30.0],
                ],
                columns=["date", "security_id", "shares_outstanding"],
            )
            fx = pd.DataFrame(
                [
                    ["1999-12-31", "JPY", 0.01],
                    ["2000-12-31", "JPY", 0.01],
                ],
                columns=["date", "currency", "fx_to_usd"],
            )
            timestamps = pd.date_range("2000-12-20 10:00:00", "2001-01-02 16:00:00", freq="6h")
            rows = []
            for i, ts in enumerate(timestamps):
                for sec, currency in [("AAA", "USD"), ("AAB", "USD"), ("BBB", "USD"), ("CCC", "JPY")]:
                    price = 100 + i
                    rows.append([ts, sec, price, price + 1, price - 1, price, price, 1000, currency, np.nan])
            bars = pd.DataFrame(rows, columns=["timestamp", "security_id", "open", "high", "low", "close", "adj_close", "volume", "currency", "fx_to_usd"])
            paths = {}
            for name, frame in [("master", master), ("prices", prices), ("shares", shares), ("fx", fx), ("bars", bars)]:
                paths[name] = root / f"{name}.parquet"
                frame.to_parquet(paths[name], index=False)
            build_cfg = root / "build.yaml"
            build_cfg.write_text(
                f"""
start: "2000-12-20"
end: "2001-01-02"
active_top_n: 2
strict_top_n: true
inputs:
  security_master: {paths['master']}
  daily_prices: {paths['prices']}
  daily_shares: {paths['shares']}
  daily_fx: {paths['fx']}
  hourly_bars: {paths['bars']}
outputs:
  security_master: {root / 'vendor' / 'security_master.parquet'}
  market_caps: {root / 'vendor' / 'market_caps.parquet'}
  hourly_bars: {root / 'vendor' / 'hourly_bars.parquet'}
column_maps:
  security_master: {{}}
  daily_prices: {{}}
  daily_shares: {{}}
  daily_fx: {{}}
  hourly_bars: {{}}
""",
                encoding="utf-8",
            )
            audit = build_canonical_from_vendor_exports(build_cfg)
            caps = pd.read_parquet(audit["outputs"]["market_caps"])
            self.assertGreater(float(caps["market_cap_usd"].max()), 0.0)
            universe = pd.read_parquet(audit["outputs"]["universe"])
            first = universe[universe["timestamp"] == universe["timestamp"].min()]
            self.assertEqual(first["company_id"].nunique(), 2)
            after_update = universe[universe["timestamp"] >= pd.Timestamp("2001-01-01 04:00:00")]
            self.assertIn("MEGA", set(after_update["company_id"]))
            mega_rows = after_update[after_update["company_id"] == "MEGA"]
            self.assertTrue((mega_rows["security_id"] == "AAA").all())


def _write_source(root: Path) -> Path:
    master = pd.DataFrame(
        [
            ["AAA", "AAA", "Alpha", "US", "Tech", "USD", "NASDAQ", "1001", True, True, "1990-01-01", ""],
            ["BBB", "BBB", "Beta", "US", "Tech", "USD", "NYSE", "1002", True, True, "1990-01-01", ""],
            ["CCC", "CCC", "Gamma", "US", "Health", "USD", "NASDAQ", "1003", True, True, "1990-01-01", ""],
        ],
        columns=[
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
        ],
    )
    caps = pd.DataFrame(
        [
            ["1999-12-31", "AAA", 100.0],
            ["1999-12-31", "BBB", 90.0],
            ["1999-12-31", "CCC", 80.0],
            ["2000-12-31", "AAA", 110.0],
            ["2000-12-31", "BBB", 70.0],
            ["2000-12-31", "CCC", 120.0],
        ],
        columns=["date", "security_id", "market_cap_usd"],
    )
    timestamps = pd.date_range("2000-12-20 10:00:00", "2001-01-04 16:00:00", freq="6h")
    rows = []
    for i, ts in enumerate(timestamps):
        for j, sec in enumerate(["AAA", "BBB", "CCC"]):
            price = 100 + i * (0.2 + j * 0.05) + j
            rows.append([ts, sec, price, price * 1.01, price * 0.99, price, price, 1_000 + i * 10 + j, 1.0])
    bars = pd.DataFrame(rows, columns=["timestamp", "security_id", "open", "high", "low", "close", "adj_close", "volume", "fx_to_usd"])
    master_path = root / "security_master.parquet"
    caps_path = root / "market_caps.parquet"
    bars_path = root / "hourly_bars.parquet"
    master.to_parquet(master_path, index=False)
    caps.to_parquet(caps_path, index=False)
    bars.to_parquet(bars_path, index=False)
    cfg_path = root / "data.yaml"
    cfg_path.write_text(
        f"""
source_label: synthetic
source_kind: vendor_point_in_time
active_top_n: 2
min_active_members: 2
inputs:
  security_master: {master_path}
  market_caps: {caps_path}
  hourly_bars: {bars_path}
outputs:
  universe: {root / 'universe.parquet'}
  features: {root / 'features.parquet'}
  audit: {root / 'audit.json'}
execution:
  commission_bps: 0.5
  half_spread_bps: 1.0
  slippage_bps: 1.0
""",
        encoding="utf-8",
    )
    return cfg_path


def _write_train(root: Path, features_path: Path) -> Path:
    path = root / "eqh10-test.yaml"
    path.write_text(
        f"""
run_name: eqh10-test
data_config: {root / 'data.yaml'}
features_path: {features_path}
save_directory: {root / 'models'}
active_top_n: 2
bars_per_year: 1638
seq_len: 4
batch_size: 4
first_train_year: 2000
test_start_years: [2001]
final_test_year: 2001
train_valid_split: 0.7
random_search_max_iterations: 1
top_n_seeds: 1
iterations: 3
early_stopping: 2
val_burnin_steps: 0
device: cpu
use_cross_attention: true
use_softmin: false
features: [r1h, r3h, z_price_1d, realized_vol_1d, rank_scaled, market_cap_log]
search_space:
  hidden_dim: [8]
  dropout: [0.0]
  lr: [0.01]
  weight_decay: [0.0]
  num_heads: [2]
""",
        encoding="utf-8",
    )
    return path


def _write_backtest(root: Path, train_cfg: Path, features_path: Path) -> Path:
    path = root / "bt.yaml"
    path.write_text(
        f"""
name: bt-eqh10-test
train_config: {train_cfg}
architecture: EQH_DEEPM_XATT
features_path: {features_path}
output_dir: {root / 'backtests'}
top_n_seeds: 1
active_top_n: 2
device: cpu
""",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
