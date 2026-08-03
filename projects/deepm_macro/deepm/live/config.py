"""Configuration loading and safety gates for daily live runs."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from deepm._paths import PROJECT_ROOT
from deepm.live.exceptions import SafetyError


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "live" / "deepm_gat_ibkr.yaml"


def load_live_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a live-trading YAML config."""
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def path_from_config(config: Mapping[str, Any], *keys: str) -> Path:
    """Read and resolve a nested path value from config."""
    value: Any = config
    for key in keys:
        value = value[key]
    return resolve_project_path(value)


def ledger_path_for_mode(config: Mapping[str, Any], mode: str) -> Path:
    """Return the ledger path for a mode without mixing fake accounts."""
    base = path_from_config(config, "paths", "ledger")
    if mode == "sim":
        return base
    suffix = mode.replace("-", "_")
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def starting_cash_for_mode(config: Mapping[str, Any], mode: str) -> float:
    """Return the configured fake-account starting equity for a mode."""
    risk = config.get("risk", {})
    if mode == "ibkr-paper" and bool(config.get("ibkr", {}).get("paper_simulation", True)):
        return float(risk.get("paper_simulator_cash", risk.get("simulator_cash", 100000.0)))
    return float(risk.get("simulator_cash", 100000.0))


def mode_provider(config: Mapping[str, Any], mode: str) -> str:
    """Return the configured data provider for a run mode."""
    data_cfg = config.get("data", {})
    if mode == "sim":
        return data_cfg.get("sim_provider", "local_parquet")
    if mode == "ibkr-paper":
        return data_cfg.get("paper_provider", "ibkr_market_data")
    if mode == "ibkr-live":
        return data_cfg.get("live_provider", "ibkr_market_data")
    raise SafetyError(f"Unknown live mode: {mode}")


def confirmation_file(config: Mapping[str, Any], run_date: date) -> Path:
    """Path to the same-day confirmation file required for live trading."""
    mode_defaults = config.get("mode_defaults", {})
    directory = resolve_project_path(mode_defaults.get("confirmation_dir", "live_confirmations"))
    return directory / f"{run_date.isoformat()}.confirm"


def assert_live_mode_allowed(config: Mapping[str, Any], mode: str, run_date: date) -> None:
    """Fail closed unless live mode is explicitly enabled and confirmed."""
    if mode != "ibkr-live":
        return

    mode_defaults = config.get("mode_defaults", {})
    if not bool(mode_defaults.get("live_enabled", False)):
        raise SafetyError("Live trading is disabled by config: mode_defaults.live_enabled is false")

    if bool(mode_defaults.get("require_confirmation_file", True)):
        expected = confirmation_file(config, run_date)
        if not expected.exists():
            raise SafetyError(
                "Live trading requires a same-day confirmation file at "
                f"{expected}"
            )


def config_for_mode(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Return a shallow-normalized copy of the config for a selected mode."""
    cfg = copy.deepcopy(dict(config))
    cfg.setdefault("mode_defaults", {})["mode"] = mode
    return cfg
