"""Config loading for the hourly equity DeePM pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = project_path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {cfg_path}")
    return data


def config_path(kind: str, name: str) -> Path:
    if name.endswith(".yaml"):
        return project_path(name)
    return PACKAGE_ROOT / "configs" / kind / f"{name}.yaml"


@dataclass(frozen=True)
class DataConfig:
    active_top_n: int
    security_master_path: Path
    market_caps_path: Path
    hourly_bars_path: Path
    universe_output_path: Path
    features_output_path: Path
    audit_output_path: Path
    source_label: str
    source_kind: str
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    min_active_members: int


def load_data_config(path: str | Path) -> DataConfig:
    raw = read_yaml(path)
    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})
    execution = raw.get("execution", {})
    active_top_n = int(raw.get("active_top_n", 10))
    return DataConfig(
        active_top_n=active_top_n,
        security_master_path=project_path(inputs.get("security_master", "")),
        market_caps_path=project_path(inputs.get("market_caps", "")),
        hourly_bars_path=project_path(inputs.get("hourly_bars", "")),
        universe_output_path=project_path(outputs.get("universe", "")),
        features_output_path=project_path(outputs.get("features", "")),
        audit_output_path=project_path(outputs.get("audit", "")),
        source_label=str(raw.get("source_label", "unknown")),
        source_kind=str(raw.get("source_kind", "vendor_point_in_time")),
        commission_bps=float(execution.get("commission_bps", 0.5)),
        half_spread_bps=float(execution.get("half_spread_bps", 1.0)),
        slippage_bps=float(execution.get("slippage_bps", 1.0)),
        min_active_members=int(raw.get("min_active_members", active_top_n)),
    )


def load_train_config(name_or_path: str) -> dict[str, Any]:
    raw = read_yaml(config_path("train", name_or_path))
    if "inherits" in raw:
        base = load_train_config(str(raw["inherits"]))
        child = {k: v for k, v in raw.items() if k != "inherits"}
        base.update(child)
        raw = base
    raw.setdefault("run_name", Path(name_or_path).stem)
    raw["data_config"] = str(project_path(raw["data_config"]))
    raw["features_path"] = str(project_path(raw["features_path"]))
    raw["save_directory"] = str(project_path(raw.get("save_directory", "equity_intraday_deepm/models")))
    raw.setdefault("bars_per_year", 1638.0)
    raw.setdefault("active_top_n", 10)
    raw.setdefault("train_valid_split", 0.9)
    raw.setdefault("test_start_years", [2010, 2015, 2020])
    raw.setdefault("final_test_year", 2026)
    raw.setdefault("first_train_year", 2000)
    raw.setdefault("top_n_seeds", 10)
    raw.setdefault("random_search_max_iterations", 50)
    raw.setdefault("iterations", 1000)
    raw.setdefault("early_stopping", 100)
    raw.setdefault("val_burnin_steps", 50)
    raw.setdefault("min_delta", 0.0)
    raw.setdefault("batch_size", 64)
    raw.setdefault("seq_len", 64)
    raw.setdefault("cost_penalty", 1.0)
    raw.setdefault("use_softmin", True)
    raw.setdefault("softmin_beta", 5.0)
    raw.setdefault("device", "auto")
    raw.setdefault("features", [])
    return raw


def load_backtest_config(name_or_path: str) -> dict[str, Any]:
    raw = read_yaml(config_path("backtest", name_or_path))
    raw["train_config"] = raw.get("train_config", raw.get("run"))
    raw["features_path"] = str(project_path(raw["features_path"]))
    raw["output_dir"] = str(project_path(raw.get("output_dir", "equity_intraday_deepm/backtests")))
    raw.setdefault("top_n_seeds", 10)
    raw.setdefault("active_top_n", 10)
    return raw
