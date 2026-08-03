"""CLI for the top-10 global stocks buy-the-move backtester."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_top10_strategy.config import load_config
from stock_top10_strategy.data import load_point_in_time_data
from stock_top10_strategy.engine import run_backtest
from stock_top10_strategy.public_data import build_wikipedia_yahoo_dataset
from stock_top10_strategy.reporting import build_report
from stock_top10_strategy.yahoo_proxy import fill_yahoo_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-10 global stocks buy-the-move strategy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate point-in-time input data")
    validate.add_argument("--config", required=True)

    backtest = subparsers.add_parser("backtest", help="Run a strategy backtest")
    backtest.add_argument("--config", required=True)
    backtest.add_argument("--run-id", default=None)

    report = subparsers.add_parser("report", help="Regenerate an HTML report for a run")
    report.add_argument("--run", required=True)

    yahoo = subparsers.add_parser(
        "fill-yahoo-prices",
        help="Fill price columns for a user-supplied point-in-time membership file",
    )
    yahoo.add_argument("--input", required=True)
    yahoo.add_argument("--output", required=True)

    wiki = subparsers.add_parser(
        "build-wikipedia-yahoo",
        help="Build a 2000-current public ranking + Yahoo price proxy dataset",
    )
    wiki.add_argument("--output", default="stock_top10_strategy/data/top10_wikipedia_yahoo_2000_current.csv")
    wiki.add_argument("--audit-output", default="stock_top10_strategy/data/top10_wikipedia_yahoo_2000_current_audit.json")
    wiki.add_argument("--mapping", default="stock_top10_strategy/configs/wikipedia_company_map.csv")
    wiki.add_argument("--start-year", type=int, default=2000)
    wiki.add_argument("--end-year", type=int, default=None)
    wiki.add_argument("--top-n", type=int, default=10)

    args = parser.parse_args()
    if args.command == "validate":
        config = load_config(args.config)
        data = load_point_in_time_data(config)
        print(f"validated rows={len(data)} dates={data['date'].nunique()} securities={data['security_id'].nunique()}")
    elif args.command == "backtest":
        config = load_config(args.config)
        run_dir = run_backtest(config, args.run_id)
        metrics_path = run_dir / "metrics.csv"
        print(f"run_dir={run_dir}")
        print(metrics_path.read_text(encoding="utf-8").strip())
    elif args.command == "report":
        report_path = build_report(Path(args.run))
        print(report_path)
    elif args.command == "fill-yahoo-prices":
        output = fill_yahoo_prices(args.input, args.output)
        print(f"{output}")
        print("warning=Yahoo prices are proxy data; membership must still be point-in-time.")
    elif args.command == "build-wikipedia-yahoo":
        output, audit = build_wikipedia_yahoo_dataset(
            output_path=args.output,
            audit_output_path=args.audit_output,
            mapping_path=args.mapping,
            start_year=args.start_year,
            end_year=args.end_year,
            top_n=args.top_n,
        )
        print(f"data={output}")
        print(f"audit={audit}")
        print("warning=public_rankings_proxy uses Wikipedia ranking snapshots and Yahoo prices; use vendor point-in-time data for production research.")


if __name__ == "__main__":
    main()
