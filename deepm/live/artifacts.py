"""Discovery helpers for completed trained DeePM artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from deepm._paths import PROJECT_ROOT
from deepm.configs.load import load_settings_for_architecture
from deepm.live.exceptions import LiveTradingError
from deepm.live.types import ModelWindow


def model_artifact_directory(train_yaml: str, architecture: str) -> Path:
    """Return the root directory used by a trained architecture."""
    settings = load_settings_for_architecture(train_yaml, architecture)
    save_directory = Path(settings["save_directory"])
    if not save_directory.is_absolute():
        save_directory = PROJECT_ROOT / save_directory
    return save_directory / settings["description"] / settings["run_name"]


def valid_year_dirs(base_dir: Path) -> list[Path]:
    """List model window folders named with four-digit years."""
    if not base_dir.exists():
        return []
    dirs = [
        child
        for child in base_dir.iterdir()
        if child.is_dir() and child.name.isdigit() and len(child.name) == 4
    ]
    return sorted(dirs, key=lambda item: int(item.name))


def read_top_run_names(window_dir: Path, top_n: int) -> list[str]:
    """Read validation-ranked run names from all_runs.csv or best_runs.csv."""
    for filename in ("all_runs.csv", "best_runs.csv"):
        path = window_dir / filename
        if path.exists():
            runs = pd.read_csv(path, index_col=0)
            run_names = [str(index) for index in runs.index.tolist()]
            return run_names[:top_n]
    return []


def has_checkpoint_for_run(window_dir: Path, run_name: str) -> bool:
    """Check whether a discovered run has a plausible checkpoint file."""
    model_dir = window_dir / "models"
    candidates = [
        model_dir / run_name,
        model_dir / f"{run_name}.pt",
        model_dir / f"{run_name}.pth",
        model_dir / f"{run_name}.ckpt",
        model_dir / f"{run_name}.tar",
    ]
    return any(path.exists() for path in candidates)


def completed_windows(base_dir: Path, top_n: int) -> list[ModelWindow]:
    """Return model windows with ranked runs and available checkpoint files."""
    windows: list[ModelWindow] = []
    year_dirs = valid_year_dirs(base_dir)
    for index, year_dir in enumerate(year_dirs):
        run_names = read_top_run_names(year_dir, top_n)
        available = [run for run in run_names if has_checkpoint_for_run(year_dir, run)]
        if not available:
            continue
        next_year: Optional[int] = None
        if index + 1 < len(year_dirs):
            next_year = int(year_dirs[index + 1].name)
        windows.append(
            ModelWindow(
                start_year=int(year_dir.name),
                end_year=next_year,
                path=year_dir,
                run_names=available[:top_n],
            )
        )
    return windows


def latest_completed_window(
    train_yaml: str,
    architecture: str,
    top_n: int,
    asof_year: int,
    base_dir: Path | None = None,
) -> ModelWindow:
    """Find the most recent completed model window usable for an as-of year."""
    root = base_dir or model_artifact_directory(train_yaml, architecture)
    candidates = [
        window
        for window in completed_windows(root, top_n)
        if window.start_year <= asof_year and (window.end_year is None or asof_year < window.end_year)
    ]
    if not candidates:
        candidates = [
            window for window in completed_windows(root, top_n) if window.start_year <= asof_year
        ]
    if not candidates:
        raise LiveTradingError(
            "No completed model artifacts found under "
            f"{root}. Wait for training to finish before running the live add-on."
        )
    return sorted(candidates, key=lambda item: item.start_year)[-1]


def assert_required_runs(window: ModelWindow, top_n: int) -> None:
    """Raise a clear error if fewer than the requested ensemble runs exist."""
    if len(window.run_names) < top_n:
        raise LiveTradingError(
            f"Model window {window.start_year} has {len(window.run_names)} checkpointed runs, "
            f"but config asks for top_n_seeds={top_n}."
        )


def summarize_windows(windows: Iterable[ModelWindow]) -> list[dict[str, object]]:
    """Small JSON-friendly summary for audit files."""
    return [
        {
            "start_year": window.start_year,
            "end_year": window.end_year,
            "path": str(window.path),
            "run_count": len(window.run_names),
            "run_names": window.run_names,
        }
        for window in windows
    ]
