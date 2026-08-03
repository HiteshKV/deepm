#!/usr/bin/env python
"""Build DeRegiME sidecar features for DeePM experiments.

Default usage:

    .venv/bin/python scripts/build_deregime_sidecar.py

This writes:
    data/deregime/deepm_deregime_input_20260625.csv
    data/deregime/deregime_sidecar_data_20260625.parquet
    data/feats-data_20260625_deregime.parquet
    data/deregime/deregime_sidecar_audit_20260625.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepm._paths import DATA_DIR, PROJECT_ROOT
from deepm.configs.load import load_train_settings
from deepm.deregime_bridge import (
    DEFAULT_HORIZONS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_NUM_REGIMES,
    DEFAULT_SEQ_LEN,
    DEFAULT_TARGET_LAG,
    build_causal_proxy_sidecar,
    compile_deregime_artifact_sidecar,
    filter_deepm_features,
    merge_sidecar_features,
    validate_sidecar_alignment,
    write_deregime_long_csv,
    write_sidecar_audit,
)


def parse_horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def snapshot_slug(path: Path) -> str:
    stem = path.name
    if stem.startswith("feats-"):
        stem = stem[len("feats-") :]
    return Path(stem).stem.replace("data_", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DeRegiME sidecar features")
    parser.add_argument(
        "--features",
        type=Path,
        default=DATA_DIR / "feats-data_20260625.parquet",
        help="Existing DeePM feature cache",
    )
    parser.add_argument(
        "--train-config",
        default="deepm-gat",
        help="DeePM train config used for ticker subset and target column",
    )
    parser.add_argument(
        "--method",
        choices=["proxy", "artifacts"],
        default="proxy",
        help="proxy builds causal rolling sidecar; artifacts compiles DeRegiME CSV outputs",
    )
    parser.add_argument(
        "--artifacts-predictions-dir",
        type=Path,
        default=None,
        help="Directory containing DeRegiME predictions__h*.csv and activations__h*.csv",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated forecast horizons to export",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--target-lag", type=int, default=DEFAULT_TARGET_LAG)
    parser.add_argument("--num-regimes", type=int, default=DEFAULT_NUM_REGIMES)
    parser.add_argument(
        "--output-feature-cache",
        type=Path,
        default=None,
        help="Merged DeePM feature cache output",
    )
    parser.add_argument("--sidecar-output", type=Path, default=None)
    parser.add_argument("--long-csv-output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    settings = load_train_settings(args.train_config)
    target_col = settings.get("target", "target")

    slug = snapshot_slug(args.features)
    base_dir = DATA_DIR / "deregime"
    long_csv = args.long_csv_output or base_dir / f"deepm_deregime_input_{slug}.csv"
    sidecar_output = args.sidecar_output or base_dir / f"deregime_sidecar_data_{slug}.parquet"
    merged_output = (
        args.output_feature_cache
        or DATA_DIR / f"feats-data_{slug}_deregime.parquet"
    )
    audit_output = args.audit_output or base_dir / f"deregime_sidecar_audit_{slug}.json"

    print(f"Reading DeePM features: {args.features}")
    features = pd.read_parquet(args.features)
    features.index = pd.to_datetime(features.index)
    features = filter_deepm_features(features, settings)

    print(f"Writing DeRegiME long input: {long_csv}")
    write_deregime_long_csv(features, long_csv, target_col=target_col)

    if args.method == "proxy":
        print("Building causal proxy sidecar features")
        sidecar = build_causal_proxy_sidecar(
            features,
            target_col=target_col,
            horizons=horizons,
            seq_len=args.seq_len,
            min_history=args.min_history,
            target_lag=args.target_lag,
            num_regimes=args.num_regimes,
        )
    else:
        if args.artifacts_predictions_dir is None:
            parser.error("--artifacts-predictions-dir is required with --method artifacts")
        print(f"Compiling DeRegiME artifacts: {args.artifacts_predictions_dir}")
        sidecar = compile_deregime_artifact_sidecar(
            args.artifacts_predictions_dir,
            horizons=horizons,
            num_regimes=args.num_regimes,
        )

    sidecar_output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_parquet(sidecar_output)
    print(f"Saved sidecar: {sidecar_output} ({sidecar.shape[0]:,} rows)")

    validation = validate_sidecar_alignment(
        features,
        sidecar,
        expected_tickers=len(settings.get("ticker_subset", [])) or None,
    )
    if validation["missing_base_pairs"]:
        raise ValueError(
            "Sidecar does not cover every DeePM feature row: "
            f"{validation['missing_base_pairs']} date/ticker pairs missing"
        )
    merged = merge_sidecar_features(
        features,
        sidecar,
        horizons=horizons,
        num_regimes=args.num_regimes,
    )
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_output)
    print(f"Saved merged feature cache: {merged_output} ({merged.shape[1]} columns)")

    write_sidecar_audit(
        audit_output,
        method=args.method,
        settings=settings,
        input_features=args.features.relative_to(PROJECT_ROOT)
        if args.features.is_relative_to(PROJECT_ROOT)
        else args.features,
        sidecar_output=sidecar_output.relative_to(PROJECT_ROOT)
        if sidecar_output.is_relative_to(PROJECT_ROOT)
        else sidecar_output,
        merged_output=merged_output.relative_to(PROJECT_ROOT)
        if merged_output.is_relative_to(PROJECT_ROOT)
        else merged_output,
        target_col=target_col,
        horizons=horizons,
        seq_len=args.seq_len,
        min_history=args.min_history,
        target_lag=args.target_lag,
        validation=validation,
    )
    print(f"Saved audit: {audit_output}")
    print("Done.")


if __name__ == "__main__":
    main()
