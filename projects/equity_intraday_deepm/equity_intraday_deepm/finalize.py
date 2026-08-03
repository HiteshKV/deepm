"""Finalize local hourly equity seed sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from equity_intraday_deepm.config import load_train_config


def finalize_run(run: str, top_n: int, allow_partial: bool = False) -> list[Path]:
    cfg = load_train_config(run)
    root = Path(cfg["save_directory"]) / cfg["run_name"]
    written: list[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"No model directory found for run {run}: {root}")
    for arch_dir in root.iterdir():
        if not arch_dir.is_dir():
            continue
        for window_dir in arch_dir.iterdir():
            all_runs_path = window_dir / "all_runs.csv"
            if not all_runs_path.exists():
                continue
            all_runs = pd.read_csv(all_runs_path, index_col=0)
            usable = []
            for run_name, row in all_runs.iterrows():
                if (window_dir / "models" / f"{run_name}.pt").exists() and (window_dir / "settings" / f"{run_name}.json").exists():
                    usable.append((run_name, row))
            if len(usable) < top_n and not allow_partial:
                raise RuntimeError(f"{window_dir} has only {len(usable)} usable runs; need {top_n}")
            frame = pd.DataFrame([row for _, row in usable], index=[name for name, _ in usable])
            frame.index.name = "run_name"
            frame = frame.sort_values("valid_loss_best", ascending=False).head(top_n)
            frame.to_csv(window_dir / "best_runs.csv")
            (window_dir / "best_runs_mean.json").write_text(
                json.dumps(frame[["valid_loss_best", "test_sharpe_net"]].mean().to_dict(), indent=2),
                encoding="utf-8",
            )
            written.append(window_dir / "best_runs.csv")
    if not written:
        raise FileNotFoundError(f"No all_runs.csv files found under {root}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finalize hourly equity sweep results")
    parser.add_argument("--run", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    for path in finalize_run(args.run, args.top_n, args.allow_partial):
        print(path)


if __name__ == "__main__":
    main()
