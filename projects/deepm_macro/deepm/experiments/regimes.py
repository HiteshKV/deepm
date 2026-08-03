"""Named experiment lanes for original, DeRegiME, and Momentum DeePM work.

The registry keeps research variants partitioned by config/artifact namespace
while still making combined comparisons easy to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


BASELINE_BACKTESTS = [
    "bt-baseline-longonly",
    "bt-baseline-tsmom",
    "bt-baseline-tsmom-rm",
    "bt-baseline-tsmom-mvo",
    "bt-baseline-tsmom-mvo-tp",
    "bt-baseline-tsmom-erc",
    "bt-baseline-macd",
    "bt-baseline-macd-rm",
    "bt-baseline-macd-mvo-tp",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """One train/backtest pair within an experiment regime."""

    name: str
    train_yaml: str
    architecture: str
    backtest_yaml: str
    top_n: int = 10
    description: str = ""
    device: str | None = None

    @property
    def train_command(self) -> str:
        return self.train_command_for()

    def train_command_for(self, filter_start_years: Iterable[int] | None = None) -> str:
        suffix = ""
        if self.device:
            suffix += f" --device {self.device}"
        if filter_start_years:
            years = " ".join(str(year) for year in filter_start_years)
            suffix += f" -fsy {years}"
        return (
            f".venv/bin/python -m deepm.training "
            f"-r {self.train_yaml} -a {self.architecture}{suffix}"
        )

    @property
    def finalize_command(self) -> str:
        return (
            ".venv/bin/python scripts/finalize_sweep_results.py "
            f"--run {self.train_yaml} --arch {self.architecture} "
            f"--top-n {self.top_n} --entity $WANDB_ENTITY --allow-partial"
        )

    @property
    def backtest_command(self) -> str:
        return (
            f".venv/bin/python -m deepm.backtest "
            f"--name {self.backtest_yaml} --diagnostics"
        )


@dataclass(frozen=True)
class Regime:
    """A distinct research lane with isolated train/backtest configs."""

    key: str
    title: str
    metrics_csv: str
    aggregate_title: str
    experiments: tuple[ExperimentSpec, ...]
    notes: str = ""

    @property
    def backtests(self) -> list[str]:
        return [experiment.backtest_yaml for experiment in self.experiments]

    def aggregate_command(self, include_baselines: bool = True) -> str:
        backtests = [*BASELINE_BACKTESTS, *self.backtests] if include_baselines else self.backtests
        joined = " \\\n  ".join(backtests)
        return (
            ".venv/bin/python scripts/aggregate_metrics.py "
            f"--title \"{self.aggregate_title}\" "
            f"--csv {self.metrics_csv} \\\n  {joined}"
        )


REGIMES: dict[str, Regime] = {
    "original": Regime(
        key="original",
        title="Original DeePM-GAT",
        metrics_csv="backtest_results/current_20260625_original_k10_metrics.csv",
        aggregate_title="Original DeePM-GAT K=10: Yahoo Proxy Data (2010-2026)",
        experiments=(
            ExperimentSpec(
                name="deepm-gat-k10",
                train_yaml="deepm-gat",
                architecture="DeePM",
                backtest_yaml="bt-deepm-gat-k10-current",
                top_n=10,
                description="Base DeePM-GAT control with K=10 seed ensemble.",
            ),
        ),
        notes="Use this lane when you want the original DeePM model without sidecar features.",
    ),
    "deregime": Regime(
        key="deregime",
        title="DeRegiME Sidecar DeePM-GAT",
        metrics_csv="backtest_results/current_20260625_deregime_metrics.csv",
        aggregate_title="DeRegiME Sidecar DeePM-GAT: Yahoo Proxy Data (2010-2026)",
        experiments=(
            ExperimentSpec(
                name="deepm-gat-dg-mean",
                train_yaml="deepm-gat-dg-mean",
                architecture="DeePM",
                backtest_yaml="bt-deepm-gat-dg-mean",
                top_n=10,
                description="Mean/probability DeRegiME feature sidecar.",
            ),
            ExperimentSpec(
                name="deepm-gat-dg-risk",
                train_yaml="deepm-gat-dg-risk",
                architecture="DeePM",
                backtest_yaml="bt-deepm-gat-dg-risk",
                top_n=10,
                description="Uncertainty/tail-risk DeRegiME feature sidecar.",
            ),
            ExperimentSpec(
                name="deepm-gat-dg-full",
                train_yaml="deepm-gat-dg-full",
                architecture="DeePM",
                backtest_yaml="bt-deepm-gat-dg-full",
                top_n=10,
                description="Full DeRegiME feature sidecar.",
            ),
        ),
        notes="Use this lane only after causal DeRegiME sidecar features have been exported.",
    ),
    "momentum": Regime(
        key="momentum",
        title="Momentum Transformer DeePM",
        metrics_csv="backtest_results/current_20260625_momentum_k10_metrics.csv",
        aggregate_title="Momentum Transformer DeePM K=10: Yahoo Proxy Data (2010-2026)",
        experiments=(
            ExperimentSpec(
                name="deepm-mt-vsn",
                train_yaml="deepm-mt-vsn",
                architecture="MT_DEEPM",
                backtest_yaml="bt-deepm-mt-vsn-k10-current",
                top_n=10,
                description="Portfolio-level Momentum Transformer with DeePM objective.",
                device="mps",
            ),
        ),
        notes="Use this lane for the Momentum Transformer integration, separate from DeRegiME.",
    ),
}


def commands_for_regime(
    regime_key: str,
    stages: Iterable[str] = ("train", "finalize", "backtest", "aggregate"),
    filter_start_years: Iterable[int] | None = None,
) -> list[str]:
    """Return shell commands for a named regime and ordered stages."""
    if regime_key not in REGIMES:
        raise KeyError(f"Unknown regime '{regime_key}'. Choose from {sorted(REGIMES)}")
    regime = REGIMES[regime_key]
    commands: list[str] = []
    for stage in stages:
        if stage == "train":
            commands.extend(
                experiment.train_command_for(filter_start_years)
                for experiment in regime.experiments
            )
        elif stage == "finalize":
            commands.extend(experiment.finalize_command for experiment in regime.experiments)
        elif stage == "backtest":
            commands.extend(experiment.backtest_command for experiment in regime.experiments)
        elif stage == "aggregate":
            commands.append(regime.aggregate_command())
        else:
            raise KeyError(
                f"Unknown stage '{stage}'. Choose from train, finalize, backtest, aggregate."
            )
    return commands


def combined_aggregate_command(regime_keys: Iterable[str] = REGIMES.keys()) -> str:
    """Return an aggregate command comparing baselines plus selected regimes."""
    chosen = [REGIMES[key] for key in regime_keys]
    backtests = [*BASELINE_BACKTESTS]
    for regime in chosen:
        backtests.extend(regime.backtests)
    joined = " \\\n  ".join(backtests)
    return (
        ".venv/bin/python scripts/aggregate_metrics.py "
        "--title \"Original vs DeRegiME vs Momentum: Yahoo Proxy Data (2010-2026)\" "
        "--csv backtest_results/current_20260625_all_regimes_metrics.csv \\\n  "
        f"{joined}"
    )
