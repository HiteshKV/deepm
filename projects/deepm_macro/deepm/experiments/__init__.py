"""Experiment-regime registry for DeePM research lanes."""

from deepm.experiments.regimes import (
    BASELINE_BACKTESTS,
    REGIMES,
    ExperimentSpec,
    Regime,
    combined_aggregate_command,
    commands_for_regime,
)

__all__ = [
    "BASELINE_BACKTESTS",
    "REGIMES",
    "ExperimentSpec",
    "Regime",
    "combined_aggregate_command",
    "commands_for_regime",
]
