#!/usr/bin/env python
"""Compile rolling DeRegiME artifacts into the DeePM sidecar feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deepm._paths import DATA_DIR, PROJECT_ROOT
from deepm.configs.load import load_train_settings
from deepm.deregime_bridge import (
    DEFAULT_HORIZONS,
    DEFAULT_NUM_REGIMES,
    compile_deregime_artifact_sidecar,
    filter_deepm_features,
    merge_sidecar_features,
    validate_sidecar_alignment,
    write_sidecar_audit,
)


def parse_horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def dataset_slug(file_path: str) -> str:
    return Path(file_path).stem.lower().replace(" ", "_")


def predictions_dirs(run_root: Path, dataset: str, experiment: str) -> list[Path]:
    base = run_root / dataset / experiment
    if not base.exists():
        return []
    return sorted(path / "predictions" for path in base.glob("seed_*") if (path / "predictions").is_dir())


def average_seed_sidecars(sidecars: list[pd.DataFrame], num_regimes: int) -> pd.DataFrame:
    if not sidecars:
        raise ValueError("No sidecars to average")
    combined = pd.concat(
        [s.assign(_seed_idx=i).reset_index(names="date") for i, s in enumerate(sidecars)],
        ignore_index=True,
    )
    numeric_cols = [
        c for c in combined.columns
        if c not in {"date", "ticker", "_seed_idx"} and pd.api.types.is_numeric_dtype(combined[c])
    ]
    averaged = combined.groupby(["date", "ticker"], as_index=False)[numeric_cols].mean()

    horizons = sorted(
        {
            int(col.rsplit("_h", 1)[1])
            for col in averaged.columns
            if col.startswith("dg_mu_h")
        }
    )
    for h in horizons:
        pi_cols = [f"dg_pi{r}_h{h}" for r in range(num_regimes) if f"dg_pi{r}_h{h}" in averaged]
        if pi_cols:
            probs = averaged[pi_cols].clip(lower=0.0)
            row_sum = probs.sum(axis=1).replace(0.0, np.nan)
            probs = probs.div(row_sum, axis=0).fillna(1.0 / len(pi_cols))
            averaged[pi_cols] = probs
            clipped = probs.clip(lower=1e-12)
            averaged[f"dg_regime_entropy_h{h}"] = (
                -(clipped * np.log(clipped)).sum(axis=1) / np.log(len(pi_cols))
            )
            averaged[f"dg_regime_maxprob_h{h}"] = probs.max(axis=1)
            averaged[f"dg_regime_id_h{h}"] = probs.to_numpy().argmax(axis=1)

    averaged["dg_available"] = 1.0
    return averaged.set_index("date").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rolling DeRegiME DeePM sidecar")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("data/deregime/deepm_deregime_rolling_experiments_20260625.json"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("deregime_runs"))
    parser.add_argument("--features", type=Path, default=DATA_DIR / "feats-data_20260625.parquet")
    parser.add_argument("--train-config", default="deepm-gat")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
    )
    parser.add_argument("--num-regimes", type=int, default=DEFAULT_NUM_REGIMES)
    parser.add_argument(
        "--sidecar-output",
        type=Path,
        default=DATA_DIR / "deregime" / "deregime_rolling_sidecar_data_20260625.parquet",
    )
    parser.add_argument(
        "--output-feature-cache",
        type=Path,
        default=DATA_DIR / "feats-data_20260625_deregime.parquet",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DATA_DIR / "deregime" / "deregime_rolling_sidecar_audit_20260625.json",
    )
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    settings = load_train_settings(args.train_config)
    experiments = json.loads(args.experiments.read_text(encoding="utf-8"))

    window_sidecars = []
    coverage = []
    for experiment in experiments:
        name = experiment["name"]
        overrides = experiment["overrides"]
        dataset = dataset_slug(overrides["file"])
        dirs = predictions_dirs(args.run_root, dataset, name)
        if not dirs:
            raise FileNotFoundError(
                f"No DeRegiME prediction directories found for {name} under "
                f"{args.run_root / dataset / name}"
            )
        seed_sidecars = [
            compile_deregime_artifact_sidecar(
                directory,
                horizons=horizons,
                num_regimes=args.num_regimes,
            )
            for directory in dirs
        ]
        sidecar = average_seed_sidecars(seed_sidecars, args.num_regimes)
        start = pd.Timestamp(overrides["test_start_date"])
        end = pd.Timestamp(overrides["test_end_date"])
        sidecar = sidecar[(pd.to_datetime(sidecar.index) >= start) & (pd.to_datetime(sidecar.index) < end)]
        window_sidecars.append(sidecar)
        coverage.append(
            {
                "experiment": name,
                "seed_prediction_dirs": [str(path) for path in dirs],
                "test_start_date": str(start.date()),
                "test_end_date": str(end.date()),
                "rows": int(len(sidecar)),
            }
        )

    sidecar_all = pd.concat(window_sidecars).sort_index()
    duplicate_mask = sidecar_all.reset_index().duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Rolling DeRegiME sidecar has duplicate date/ticker rows")

    args.sidecar_output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_all.to_parquet(args.sidecar_output)

    features = pd.read_parquet(args.features)
    features.index = pd.to_datetime(features.index)
    features = filter_deepm_features(features, settings)
    validation = validate_sidecar_alignment(features, sidecar_all)
    merged = merge_sidecar_features(
        features,
        sidecar_all,
        horizons=horizons,
        num_regimes=args.num_regimes,
    )
    args.output_feature_cache.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_feature_cache)

    write_sidecar_audit(
        args.audit_output,
        method="artifacts_rolling",
        settings=settings,
        input_features=args.features.relative_to(PROJECT_ROOT)
        if args.features.is_relative_to(PROJECT_ROOT)
        else args.features,
        sidecar_output=args.sidecar_output.relative_to(PROJECT_ROOT)
        if args.sidecar_output.is_relative_to(PROJECT_ROOT)
        else args.sidecar_output,
        merged_output=args.output_feature_cache.relative_to(PROJECT_ROOT)
        if args.output_feature_cache.is_relative_to(PROJECT_ROOT)
        else args.output_feature_cache,
        target_col=settings.get("target", "target"),
        horizons=horizons,
        seq_len=int(experiments[0]["overrides"].get("seq_len", 252)),
        min_history=0,
        target_lag=0,
        validation={**validation, "rolling_coverage": coverage},
        leakage_policy=(
            "Rolling artifact sidecar: each DeRegiME experiment trains/validates "
            "strictly before its test_start_date, then emits predictions only "
            "inside that DeePM test window. Rows without rolling artifacts are "
            "filled neutral with dg_available=0."
        ),
    )
    print(f"Saved rolling sidecar: {args.sidecar_output} ({len(sidecar_all):,} rows)")
    print(f"Saved merged DeePM feature cache: {args.output_feature_cache}")
    print(f"Saved audit: {args.audit_output}")


if __name__ == "__main__":
    main()
