#!/usr/bin/env python
"""Audit W&B/local artifacts before resuming a DeePM training run."""

import argparse
import os
import sys
from pathlib import Path

import wandb

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from deepm._paths import PROJECT_ROOT
from deepm.configs.load import load_settings_for_architecture, load_sweep_settings
from deepm.configs.settings import WANDB_ENTITY

SUMMARY_KEYS = [
    "valid_loss_best",
    "test_sharpe_gross",
    "test_sharpe_net",
    "test_calmar_gross",
    "test_calmar_net",
]


def model_root(settings: dict) -> Path:
    save_directory = Path(settings["save_directory"])
    if not save_directory.is_absolute():
        save_directory = PROJECT_ROOT / save_directory
    return save_directory / settings["description"] / settings["run_name"]


def has_local_artifacts(window_dir: Path, run_name: str) -> bool:
    return all(
        path.exists()
        for path in (
            window_dir / "models" / run_name,
            window_dir / "settings" / f"{run_name}.json",
            window_dir / "data-params" / f"{run_name}.pkl",
        )
    )


def search_space_size(sweep_settings: dict) -> int | None:
    size = 1
    for spec in sweep_settings.get("parameters", {}).values():
        values = spec.get("values")
        if values is None:
            return None
        size *= len(values)
    return size


def audit_window(
    api,
    entity: str,
    settings: dict,
    sweep_settings: dict,
    architecture: str,
    test_start: int,
) -> dict:
    group = f"{settings['description']}_{architecture}_{test_start}"
    path = f"{entity}/{settings['project']}"
    runs = list(api.runs(path, filters={"group": group}, per_page=10000))
    window_dir = model_root(settings) / str(test_start)
    states = {}
    sweep_counts = {}
    usable = 0
    complete_summary = 0
    for run in runs:
        states[run.state] = states.get(run.state, 0) + 1
        if run.sweep is not None:
            sweep_counts[run.sweep.id] = sweep_counts.get(run.sweep.id, 0) + 1
        has_summary = all(key in run.summary._json_dict for key in SUMMARY_KEYS)
        if has_summary:
            complete_summary += 1
        if run.state == "finished" and has_summary and has_local_artifacts(window_dir, run.name):
            usable += 1

    local_models = len(list((window_dir / "models").glob("*"))) if (window_dir / "models").exists() else 0
    local_settings = (
        len(list((window_dir / "settings").glob("*.json")))
        if (window_dir / "settings").exists()
        else 0
    )
    required = int(settings["random_search_max_iterations"])
    space_size = search_space_size(sweep_settings)
    return {
        "test_start": test_start,
        "group": group,
        "required": required,
        "usable": usable,
        "remaining": max(0, required - usable),
        "wandb_total": len(runs),
        "wandb_states": states,
        "sweep_space_size": space_size,
        "sweep_counts": sweep_counts,
        "complete_summaries": complete_summary,
        "local_models": local_models,
        "local_settings": local_settings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit training resume state")
    parser.add_argument("-r", "--run-file-name", default="deepm-gat")
    parser.add_argument("-a", "--arch", default="DeePM")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or WANDB_ENTITY)
    args = parser.parse_args()

    if not args.entity:
        raise SystemExit("Set WANDB_ENTITY or pass --entity")

    settings = load_settings_for_architecture(args.run_file_name, args.arch)
    sweep_settings = load_sweep_settings(settings["sweep_yaml"])
    api = wandb.Api()
    rows = [
        audit_window(api, args.entity, settings, sweep_settings, args.arch, int(test_start))
        for test_start in settings["test_start_years"]
    ]

    print(f"Training resume audit for {args.run_file_name} / {args.arch}")
    print(f"W&B path: {args.entity}/{settings['project']}")
    print("")
    for row in rows:
        print(
            f"{row['test_start']}: usable {row['usable']}/{row['required']} "
            f"(remaining {row['remaining']}), local models/settings "
            f"{row['local_models']}/{row['local_settings']}, W&B states {row['wandb_states']}"
        )
        if row["sweep_space_size"] is not None:
            exhausted = [
                sweep_id
                for sweep_id, count in row["sweep_counts"].items()
                if count >= row["sweep_space_size"]
            ]
            print(
                f"  sweep grid size {row['sweep_space_size']}; "
                f"sweep job counts {row['sweep_counts']}"
            )
            if exhausted and row["remaining"] > 0:
                print(
                    "  note: at least one sweep is exhausted; restart will create "
                    "a continuation sweep if needed."
                )
    remaining_windows = [str(row["test_start"]) for row in rows if row["remaining"] > 0]
    print("")
    if remaining_windows:
        print("Recommended resume filter:")
        print("  -fsy " + " ".join(remaining_windows))
    else:
        print("All windows have the configured number of usable completed runs.")


if __name__ == "__main__":
    main()
