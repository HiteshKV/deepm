#!/usr/bin/env python
"""Create rolling-window DeRegiME experiments aligned to DeePM windows.

The generated JSON trains one DeRegiME experiment per DeePM test window:

* 2010 window: train/valid dates are strictly before 2010-01-01.
* 2015 window: train/valid dates are strictly before 2015-01-01.
* 2020 window: train/valid dates are strictly before 2020-01-01.

The DeRegiME data loader receives ``test_start_date``/``test_end_date`` and
adds only a pre-test context overlap to the test loader. That context is not
used for fitting scalers, training, or validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_OVERRIDES = {
    "file": "data/deregime/deepm_deregime_input_20260625.csv",
    "date_col": "date",
    "series_id_col": "cols",
    "value_col": "data",
    "freq": "b",
    "seq_len": 252,
    "pred_len": 63,
    "gp_label_len": 48,
    "encoder_type": "patchtst",
    "model_type": "regime_gp",
    "gating_method": "stick_breaking",
    "num_regimes": 4,
    "Rmax": 8,
    "use_student_t_likelihood": True,
    "student_t_regime_mixture_likelihood": True,
    "use_residual_observation_variance": True,
    "use_residual_mle_backbone": True,
    "future_decoder_type": "sequence_flatten",
    "training_iterations": 1000,
    "min_epochs": 50,
    "patience": 50,
    "checkpoint_every": 1,
    "batch_size": 64,
    "eval_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "num_workers": 0,
    "persistent_workers": False,
    "prefetch_factor": 2,
    "pin_memory": False,
    "tensorize_dataset": True,
    "device": "auto",
    "allow_device_fallback": False,
    "plot_multi_horizons": [1, 5, 21, 63],
    "seeds": [42, 123, 456],
}

DEFAULT_TEST_START_YEARS = [2010, 2015, 2020]
DEFAULT_FINAL_TEST_YEAR = 2026


def load_base_experiment(path: Path | None) -> dict:
    """Load the first experiment from JSON or return the default template."""
    if path is None:
        return {
            "name": "deepm_full_deregime_20260625",
            "overrides": deepcopy(DEFAULT_BASE_OVERRIDES),
        }
    experiments = json.loads(path.read_text(encoding="utf-8"))
    if not experiments:
        raise ValueError(f"No experiments found in {path}")
    experiment = deepcopy(experiments[0])
    experiment.setdefault("overrides", {})
    return experiment


def load_window_settings(path: Path | None) -> dict[str, Any]:
    """Load optional DeePM-style rolling windows without importing DeePM."""
    settings = {
        "test_start_years": DEFAULT_TEST_START_YEARS,
        "final_test_year": DEFAULT_FINAL_TEST_YEAR,
    }
    if path is None:
        return settings
    if not path.exists():
        raise FileNotFoundError(f"Window config not found: {path}")
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("test_start_years", "final_test_year"):
        if key in raw:
            settings[key] = raw[key]
    return settings


def build_windows(settings: dict[str, Any]) -> list[tuple[int, int]]:
    starts = [int(value) for value in settings["test_start_years"]]
    ends = starts[1:] + [int(settings["final_test_year"]) + 1]
    return list(zip(starts, ends))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build rolling DeRegiME experiments for DeePM windows"
    )
    parser.add_argument(
        "--base-experiments",
        type=Path,
        default=Path("data/deregime/deepm_deregime_experiments_20260625.json"),
        help="Single-window DeRegiME JSON to clone. If missing, defaults are used.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help=(
            "Optional DeePM-style YAML path whose test_start_years and "
            "final_test_year should be mirrored. Defaults to 2010/2015/2020 "
            "and final_test_year 2026."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/deregime/deepm_deregime_rolling_experiments_20260625.json"),
    )
    args = parser.parse_args()

    base_path = args.base_experiments if args.base_experiments.exists() else None
    base = load_base_experiment(base_path)
    base_name = base.get("name", "deepm_full_deregime_20260625")
    base_overrides = {**deepcopy(DEFAULT_BASE_OVERRIDES), **base.get("overrides", {})}
    settings = load_window_settings(Path(args.train_config) if args.train_config else None)

    experiments = []
    for test_start, test_end in build_windows(settings):
        overrides = deepcopy(base_overrides)
        overrides.update(
            {
                "test_start_date": f"{test_start}-01-01",
                "test_end_date": f"{test_end}-01-01",
                "deepm_train_config": args.train_config,
                "deepm_test_start_year": test_start,
                "deepm_test_end_year_exclusive": test_end,
                "sidecar_protocol": "rolling_pre_test_only",
            }
        )
        experiments.append(
            {
                "name": f"{base_name}_w{test_start}",
                "overrides": overrides,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(experiments, indent=2), encoding="utf-8")
    print(f"Wrote {len(experiments)} rolling DeRegiME experiments to {args.output}")
    for exp in experiments:
        overrides = exp["overrides"]
        print(
            f"  {exp['name']}: train/valid < {overrides['test_start_date']}, "
            f"test {overrides['test_start_date']} <= date < {overrides['test_end_date']}"
        )


if __name__ == "__main__":
    main()
