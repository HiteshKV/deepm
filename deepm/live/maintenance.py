"""Model/data freshness checks for live DeePM management."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from deepm.live.artifacts import latest_completed_window
from deepm.configs.load import load_settings_for_architecture
from deepm.live.config import DEFAULT_CONFIG, load_live_config
from deepm.live.data_update import latest_snapshot_path


def _latest_price_date(path: Path) -> date | None:
    if not path.exists():
        return None
    data = pd.read_parquet(path)
    if data.empty:
        return None
    return pd.Timestamp(data.index.max()).date()


def model_update_status(config: dict[str, Any], run_date: date) -> dict[str, Any]:
    """Return a JSON-friendly model/data maintenance status."""
    model_cfg = config["model"]
    top_n = int(model_cfg["top_n_seeds"])
    train_settings = load_settings_for_architecture(
        str(model_cfg["train_yaml"]),
        str(model_cfg["architecture"]),
    )
    window = latest_completed_window(
        train_yaml=str(model_cfg["train_yaml"]),
        architecture=str(model_cfg["architecture"]),
        top_n=top_n,
        asof_year=run_date.year,
    )
    snapshot = latest_snapshot_path(config)
    latest_data_date = _latest_price_date(snapshot)

    update_cfg = config.get("model_update", {})
    max_age_days = int(update_cfg.get("max_model_age_days", 180))
    min_new_days = int(update_cfg.get("min_new_business_days_for_retrain", 60))
    final_test_year = int(train_settings.get("final_test_year", run_date.year))
    coverage_end = date(final_test_year, 12, 31)
    days_after_coverage = max(0, (run_date - coverage_end).days)

    new_business_days = None
    if latest_data_date is not None and latest_data_date > coverage_end:
        new_business_days = len(pd.bdate_range(coverage_end, latest_data_date)) - 1

    retrain_due = days_after_coverage >= max_age_days and (
        new_business_days is None or new_business_days >= min_new_days
    )
    return {
        "run_date": run_date.isoformat(),
        "latest_data_date": None if latest_data_date is None else latest_data_date.isoformat(),
        "model_train_yaml": model_cfg["train_yaml"],
        "architecture": model_cfg["architecture"],
        "top_n_seeds": top_n,
        "active_model_window": window.start_year,
        "configured_final_test_year": final_test_year,
        "configured_coverage_end": coverage_end.isoformat(),
        "active_model_path": str(window.path),
        "selected_run_count": len(window.run_names),
        "days_after_configured_coverage": days_after_coverage,
        "new_business_days_after_coverage": new_business_days,
        "policy": update_cfg.get("policy", "daily_inference_quarterly_challenger"),
        "auto_retrain_enabled": bool(update_cfg.get("auto_retrain_enabled", False)),
        "auto_promote_enabled": bool(update_cfg.get("auto_promote_enabled", False)),
        "retrain_due": retrain_due,
        "recommended_action": (
            "train_challenger_and_paper_trade_before_promotion"
            if retrain_due
            else "keep_current_model_for_daily_inference"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DeePM live model/data freshness")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_live_config(args.config)
    payload = model_update_status(config, date.fromisoformat(args.date))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
