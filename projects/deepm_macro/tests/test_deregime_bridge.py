import unittest

import numpy as np
import pandas as pd

from deepm.configs.load import load_backtest_settings, load_train_settings
from deepm.deregime_bridge import (
    build_causal_proxy_sidecar,
    feature_group_columns,
    merge_sidecar_features,
    regime_feature_columns,
    validate_sidecar_alignment,
)


class DeRegimeBridgeTests(unittest.TestCase):
    def _features(self, n=180):
        dates = pd.date_range("2000-01-03", periods=n, freq="B")
        frames = []
        for ticker, offset in [("A", 0.0), ("B", 0.1)]:
            target = np.sin(np.arange(n) / 10.0) + offset
            frame = pd.DataFrame(
                {
                    "ticker": ticker,
                    "target": target,
                    "r1d": target,
                    "vs_factor": 1.0,
                },
                index=dates,
            )
            frames.append(frame)
        return pd.concat(frames).sort_index()

    def test_proxy_sidecar_does_not_use_same_row_target(self):
        base = self._features()
        shocked = base.copy()
        last_date = shocked.index.max()
        shocked.loc[(shocked.index == last_date) & (shocked["ticker"] == "A"), "target"] = 999.0

        kwargs = {
            "horizons": (1,),
            "seq_len": 12,
            "min_history": 6,
            "target_lag": 1,
        }
        sidecar_base = build_causal_proxy_sidecar(base, **kwargs)
        sidecar_shocked = build_causal_proxy_sidecar(shocked, **kwargs)

        lhs = sidecar_base.loc[
            (sidecar_base.index == last_date) & (sidecar_base["ticker"] == "A"),
            "dg_mu_h1",
        ].iloc[0]
        rhs = sidecar_shocked.loc[
            (sidecar_shocked.index == last_date) & (sidecar_shocked["ticker"] == "A"),
            "dg_mu_h1",
        ].iloc[0]
        self.assertAlmostEqual(lhs, rhs)

    def test_merge_sidecar_features_preserves_rows_and_fills_columns(self):
        features = self._features()
        sidecar = build_causal_proxy_sidecar(
            features,
            horizons=(1,),
            seq_len=12,
            min_history=6,
            target_lag=1,
        )
        merged = merge_sidecar_features(features, sidecar, horizons=(1,))
        validation = validate_sidecar_alignment(features, sidecar, expected_tickers=2)

        self.assertEqual(len(merged), len(features))
        self.assertIn("dg_available", merged.columns)
        self.assertIn("dg_pi3_h1", merged.columns)
        self.assertEqual(validation["missing_base_pairs"], 0)
        self.assertFalse(merged[regime_feature_columns((1,))].isna().any().any())

    def test_deregime_configs_inherit_base_settings(self):
        base_backtest = load_backtest_settings("bt-deepm-gat")
        train = load_train_settings("deepm-gat-dg-risk")
        backtest = load_backtest_settings("bt-deepm-gat-dg-risk")

        self.assertEqual(train["data_parquet"], "data_20260625_deregime.parquet")
        self.assertEqual(train["final_test_year"], 2026)
        self.assertEqual(len(train["ticker_subset"]), 50)
        self.assertTrue(set(feature_group_columns("risk")).issubset(train["features"]))
        self.assertEqual(backtest["model"]["class"], "DeepMomentum")
        self.assertEqual(backtest["model"]["train_yaml"], "deepm-gat-dg-risk")
        self.assertEqual(backtest["universe"], base_backtest["universe"])


if __name__ == "__main__":
    unittest.main()
