#!/usr/bin/env python
"""Finalize usable DeePM sweep artifacts for seed-ensemble backtests."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from deepm._paths import PROJECT_ROOT
from deepm.configs.load import load_settings_for_architecture
from deepm.configs.settings import WANDB_ENTITY

SUMMARY_KEYS = [
    "valid_loss_best",
    "test_sharpe_gross",
    "test_sharpe_net",
    "test_calmar_gross",
    "test_calmar_net",
]


def model_root(settings: dict[str, Any]) -> Path:
    save_directory = Path(settings["save_directory"])
    if not save_directory.is_absolute():
        save_directory = PROJECT_ROOT / save_directory
    return save_directory / settings["description"] / settings["run_name"]


def report_dir_for(settings: dict[str, Any], top_n: int) -> Path:
    snapshot = Path(settings["data_parquet"]).stem.replace("data_", "")
    return PROJECT_ROOT / "reports" / f"{settings['description'].replace('-', '_')}_k{top_n}_{snapshot}"


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_settings_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def local_metric_row(settings_json: dict[str, Any]) -> dict[str, float]:
    return {
        "valid_loss_best": _finite_or_nan(
            settings_json.get("valid_loss_best", settings_json.get("valid_sharpe"))
        ),
        "test_sharpe_gross": _finite_or_nan(
            settings_json.get("test_sharpe_gross", settings_json.get("test_sharpe"))
        ),
        "test_sharpe_net": _finite_or_nan(settings_json.get("test_sharpe_net")),
        "test_calmar_gross": _finite_or_nan(
            settings_json.get("test_calmar_gross", settings_json.get("test_calmar"))
        ),
        "test_calmar_net": _finite_or_nan(settings_json.get("test_calmar_net")),
    }


def wandb_summaries(
    entity: str | None,
    project: str,
    group: str,
) -> dict[str, dict[str, Any]]:
    if not entity:
        return {}
    try:
        import wandb
    except ImportError:
        return {}

    try:
        runs = wandb.Api().runs(f"{entity}/{project}", filters={"group": group}, per_page=10000)
        return {run.name: dict(run.summary._json_dict) for run in runs}
    except Exception as exc:  # pragma: no cover - depends on W&B service availability
        print(f"Warning: could not read W&B summaries for {group}: {exc}", file=sys.stderr)
        return {}


def usable_runs(window_dir: Path, summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.Series] = []
    settings_dir = window_dir / "settings"
    for settings_path in sorted(settings_dir.glob("*.json")):
        run_name = settings_path.stem
        required = [
            window_dir / "models" / run_name,
            settings_path,
            window_dir / "data-params" / f"{run_name}.pkl",
        ]
        if not all(path.exists() for path in required):
            continue

        settings_json = load_settings_json(settings_path)
        row = local_metric_row(settings_json)
        summary = summaries.get(run_name, {})
        for key in SUMMARY_KEYS:
            if key in summary and not pd.isna(summary[key]):
                row[key] = _finite_or_nan(summary[key])

        if pd.isna(row["valid_loss_best"]):
            continue
        rows.append(pd.Series(row, name=run_name))

    if not rows:
        return pd.DataFrame(columns=SUMMARY_KEYS)
    frame = pd.concat(rows, axis=1).T
    return frame[SUMMARY_KEYS].sort_values("valid_loss_best", ascending=False)


def write_window_outputs(window_dir: Path, ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    best = ranked.head(top_n)
    ranked.to_csv(window_dir / "all_runs.csv")
    best.to_csv(window_dir / "best_runs.csv")
    best.mean(numeric_only=True).to_json(window_dir / "best_runs_mean.json", indent=4)
    return best


def finalize(
    run: str,
    arch: str,
    top_n: int,
    entity: str | None,
    allow_partial: bool,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = load_settings_for_architecture(run, arch)
    root = model_root(settings)
    output_dir = output_dir or report_dir_for(settings, top_n)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = []
    summary_rows = []
    for test_start in settings["test_start_years"]:
        window_dir = root / str(test_start)
        group = f"{settings['description']}_{arch}_{test_start}"
        summaries = wandb_summaries(entity, settings["project"], group)
        ranked = usable_runs(window_dir, summaries)
        if len(ranked) < top_n:
            raise SystemExit(
                f"{test_start} has only {len(ranked)} usable runs; need top_n={top_n}."
            )
        if not allow_partial and len(ranked) < int(settings["random_search_max_iterations"]):
            raise SystemExit(
                f"{test_start} has {len(ranked)} usable runs; expected "
                f"{settings['random_search_max_iterations']}. Use --allow-partial to continue."
            )

        best = write_window_outputs(window_dir, ranked, top_n)
        summary_rows.append(
            {
                "test_start": test_start,
                "usable_runs": len(ranked),
                "selected_runs": len(best),
                "best_valid_sharpe": float(best["valid_loss_best"].iloc[0]),
                "mean_selected_valid_sharpe": float(best["valid_loss_best"].mean()),
            }
        )
        for rank, (run_name, row) in enumerate(best.iterrows(), start=1):
            selection_rows.append(
                {
                    "test_start": test_start,
                    "rank": rank,
                    "run_name": run_name,
                    **{key: row.get(key) for key in SUMMARY_KEYS},
                }
            )

    selection = pd.DataFrame(selection_rows)
    summary = pd.DataFrame(summary_rows)
    selection.to_csv(output_dir / "window_model_selection.csv", index=False)
    summary.to_csv(output_dir / "selection_summary.csv", index=False)
    return selection, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize DeePM sweep results for backtesting")
    parser.add_argument("--run", default="deepm-gat")
    parser.add_argument("--arch", default="DeePM")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or WANDB_ENTITY)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection, summary = finalize(
        run=args.run,
        arch=args.arch,
        top_n=args.top_n,
        entity=args.entity,
        allow_partial=args.allow_partial,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print("Selection summary:")
    print(summary.to_string(index=False))
    print("")
    print("Selected runs:")
    print(selection[["test_start", "rank", "run_name", "valid_loss_best"]].to_string(index=False))


if __name__ == "__main__":
    main()
