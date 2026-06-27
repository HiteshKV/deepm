"""Trained model discovery and daily signal generation."""

from __future__ import annotations

from datetime import date

import pandas as pd

from deepm.backtest.models.deep_momentum import DeepMomentum
from deepm.live.artifacts import assert_required_runs, latest_completed_window
from deepm.live.exceptions import DataUnavailable, LiveTradingError
from deepm.live.types import ModelWindow, Signal


class SignalEngine:
    """Generate model positions from the latest completed DeePM ensemble."""

    def __init__(self, model_config: dict[str, object]):
        self.model_config = model_config

    def discover_window(self, run_date: date) -> ModelWindow:
        top_n = int(self.model_config["top_n_seeds"])
        window = latest_completed_window(
            train_yaml=str(self.model_config["train_yaml"]),
            architecture=str(self.model_config["architecture"]),
            top_n=top_n,
            asof_year=run_date.year,
        )
        assert_required_runs(window, top_n)
        return window

    @staticmethod
    def _map_features_for_model(
        features: pd.DataFrame,
        tickers: list[str],
        train_settings: dict[str, object],
    ) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
        """Apply the training ticker mapping for model inference only."""
        mapping = train_settings.get("ticker_mapping") or {}
        if not mapping:
            return features, tickers, {}

        mapped = features.copy()
        mapped["ticker"] = mapped["ticker"].map(mapping)
        mapped = mapped[mapped["ticker"].notna()].copy()
        model_tickers = [mapping[ticker] for ticker in tickers if ticker in mapping]
        reverse_mapping = {str(mapped_name): str(short_name) for short_name, mapped_name in mapping.items()}
        return mapped, model_tickers, reverse_mapping

    def generate(self, features: pd.DataFrame, tickers: list[str], run_date: date) -> tuple[list[Signal], ModelWindow]:
        """Run the trained ensemble and return latest positions by ticker."""
        if features.empty:
            raise DataUnavailable("Feature data is empty")

        window = self.discover_window(run_date)
        model = DeepMomentum(
            train_yaml=str(self.model_config["train_yaml"]),
            architecture=str(self.model_config["architecture"]),
            top_n_seeds=int(self.model_config["top_n_seeds"]),
            seq_len=int(self.model_config["seq_len"]),
            pre_loss_steps=int(self.model_config["pre_loss_steps"]),
            batch_size=int(self.model_config.get("batch_size", 8)),
        )
        model_features, model_tickers, reverse_mapping = self._map_features_for_model(
            features,
            tickers,
            model.train_settings,
        )
        if not model_tickers:
            raise DataUnavailable("No configured instruments match the trained model ticker mapping")

        start_date = f"{window.start_year}-01-01"
        try:
            predictions = model.predict(model_features, model_tickers, start_date=start_date)
        except Exception as exc:
            raise LiveTradingError(f"Failed to generate live DeePM signals: {exc}") from exc

        predictions.index = pd.to_datetime(predictions.index)
        if reverse_mapping:
            predictions = predictions.rename(columns=reverse_mapping)
        eligible = predictions.loc[predictions.index <= pd.Timestamp(run_date)]
        if eligible.empty:
            raise DataUnavailable(
                f"No predictions available on or before {run_date.isoformat()} "
                f"from window {window.start_year}"
            )
        latest = eligible.sort_index().iloc[-1].dropna()
        signals = [
            Signal(run_date=run_date, ticker=str(ticker), model_position=float(value))
            for ticker, value in latest.items()
            if ticker in tickers
        ]
        if not signals:
            raise DataUnavailable("The model returned no live positions for configured instruments")
        return signals, window
