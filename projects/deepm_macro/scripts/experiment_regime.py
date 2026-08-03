#!/usr/bin/env python
"""Print commands for isolated DeePM experiment regimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from deepm.experiments.regimes import (  # noqa: E402
    REGIMES,
    combined_aggregate_command,
    commands_for_regime,
)


def _print_regime(regime_key: str) -> None:
    regime = REGIMES[regime_key]
    print(f"{regime.key}: {regime.title}")
    print(f"  metrics: {regime.metrics_csv}")
    print(f"  notes: {regime.notes}")
    for experiment in regime.experiments:
        print(
            f"  - {experiment.name}: train={experiment.train_yaml}, "
            f"arch={experiment.architecture}, backtest={experiment.backtest_yaml}, "
            f"top_n={experiment.top_n}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available regimes")

    show_parser = subparsers.add_parser("show", help="Show one regime")
    show_parser.add_argument("regime", choices=sorted(REGIMES))

    commands_parser = subparsers.add_parser("commands", help="Print run commands")
    commands_parser.add_argument("regime", choices=sorted(REGIMES))
    commands_parser.add_argument(
        "--stage",
        choices=["train", "finalize", "backtest", "aggregate"],
        action="append",
        help="Stage to include; repeat for multiple. Defaults to all stages.",
    )
    commands_parser.add_argument(
        "-fsy",
        "--filter-start-years",
        type=int,
        nargs="+",
        help="Optional rolling test window filter for printed training commands.",
    )

    combined_parser = subparsers.add_parser(
        "combined", help="Print combined aggregate command"
    )
    combined_parser.add_argument(
        "--regime",
        choices=sorted(REGIMES),
        action="append",
        help="Regime to include; repeat for multiple. Defaults to all regimes.",
    )

    args = parser.parse_args()

    if args.command == "list":
        for key in sorted(REGIMES):
            _print_regime(key)
        return

    if args.command == "show":
        _print_regime(args.regime)
        return

    if args.command == "commands":
        stages = args.stage or ["train", "finalize", "backtest", "aggregate"]
        for command in commands_for_regime(
            args.regime,
            stages,
            filter_start_years=args.filter_start_years,
        ):
            print(command)
        return

    if args.command == "combined":
        print(combined_aggregate_command(args.regime or REGIMES.keys()))
        return

    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
