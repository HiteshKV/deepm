"""Top-level CLI for the hourly equity DeePM pilot."""

from __future__ import annotations

import sys

from equity_intraday_deepm.benchmark import main as benchmark_main
from equity_intraday_deepm.data import data_cli
from equity_intraday_deepm.finalize import main as finalize_main
from equity_intraday_deepm.vendor_import import main as vendor_main


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: python -m equity_intraday_deepm {data,benchmark-device,finalize,vendor} ...")
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "data":
        data_cli(rest)
    elif command == "benchmark-device":
        benchmark_main(rest)
    elif command == "finalize":
        finalize_main(rest)
    elif command == "vendor":
        vendor_main(rest)
    else:
        print(f"unknown command: {command}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
