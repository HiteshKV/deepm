#!/usr/bin/env python
"""Export feature and temporal-attention diagnostics for MT_DEEPM checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from deepm._paths import DATA_DIR, PROJECT_ROOT  # noqa: E402
from deepm.backtest.models.deep_momentum import (  # noqa: E402
    DeepMomentum,
    FIELDS_REQUIRED_FOR_PREDICT,
    TradingSignalDmn,
)
from deepm.configs.load import load_backtest_settings  # noqa: E402
from deepm.data.dataset import unpack_torch_dataset  # noqa: E402
from deepm.models.base import DmnMode  # noqa: E402
from deepm.training.train import TrainDeepMomentumNetwork, device  # noqa: E402


def _feature_names(settings: dict) -> list[str]:
    names = list(settings["features"])
    if settings.get("use_transaction_costs") and settings.get("tcost_inputs"):
        names.append("vs_factor_tcost_input")
    names.append("mask_single_date")
    return names


def export_interpretability(
    backtest_name: str,
    output_dir: Path,
    start_year: int | None,
    seed_rank: int,
    max_batches: int,
) -> Path:
    configs = load_backtest_settings(backtest_name)
    model_cfg = dict(configs["model"])
    model_cfg.pop("module", None)
    model_cfg.pop("class", None)
    if model_cfg["architecture"] != "MT_DEEPM":
        raise SystemExit(
            f"{backtest_name} uses architecture={model_cfg['architecture']}; "
            "expected MT_DEEPM."
        )

    deep_model = DeepMomentum(**model_cfg)
    data = pd.read_parquet(DATA_DIR / ("feats-" + configs["data_parquet"]))
    ticker_mapping = configs.get("ticker_mapping", {})
    data["ticker"] = data["ticker"].map(ticker_mapping)
    corr_feat, date_time_embedding = deep_model._build_global_features(data)
    tickers = [ticker_mapping.get(ticker, ticker) for ticker in configs["universe"]]

    years = list(deep_model.model_start_years.astype(int))
    if start_year is not None:
        years = [start_year]

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_frames = []
    attention_frames = []

    for year in years:
        all_runs = deep_model._load_seed_runs(year)
        if seed_rank >= len(all_runs):
            raise SystemExit(f"seed_rank={seed_rank} unavailable for {year}")
        run_name = str(all_runs.index[seed_rank])
        next_years = deep_model.model_start_years[
            deep_model.model_start_years.astype(int) > year
        ]
        end_year = int(next_years.iloc[0]) if len(next_years) else 9999
        tickers_dict = deep_model._read_tickers_dict(year, run_name)
        test_data = deep_model._build_test_dataset(
            data,
            tickers,
            tickers_dict,
            year,
            end_year,
            corr_feat,
            date_time_embedding,
        )

        run_settings = pd.read_json(
            Path(deep_model.get_model_directory(year)) / "settings" / f"{run_name}.json",
            typ="series",
        )
        run_settings["num_tickers"] = len(run_settings["tickers_dict"])
        predict_kwargs = {
            k: run_settings[k]
            for k in FIELDS_REQUIRED_FOR_PREDICT
            if k in run_settings
        }
        dmn = TradingSignalDmn(
            model_cfg["architecture"],
            test_data,
            num_features=len(deep_model.train_settings["features"]),
            model_save_path=deep_model.get_model_directory(year),
            parent_class=TrainDeepMomentumNetwork,
            **deep_model.train_settings,
            **predict_kwargs,
        )
        state = torch.load(
            dmn.parent.model_save_path(run_name),
            map_location=device,
        )
        dmn.model.load_state_dict(state)
        dmn.model.eval()
        dmn.model.record_temporal_attention = True

        feature_sum = None
        feature_count = 0
        attention_sum = None
        attention_count = 0

        loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=model_cfg["batch_size"],
            drop_last=False,
        )
        with torch.no_grad():
            for batch_idx, samples in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                torch_dataset = unpack_torch_dataset(
                    samples,
                    test_data,
                    device,
                    use_dates_mask=True,
                    live_mode=False,
                )
                _ = dmn.model(
                    **torch_dataset,
                    mode=DmnMode.LIVE,
                    return_temporal_attention=True,
                )
                importance = dmn.model.variable_importance(**torch_dataset)
                importance = importance.detach().cpu()
                batch_feature_sum = importance.sum(dim=(0, 1, 2))
                feature_sum = (
                    batch_feature_sum
                    if feature_sum is None
                    else feature_sum + batch_feature_sum
                )
                feature_count += importance.shape[0] * importance.shape[1] * importance.shape[2]

                attention = dmn.model.last_temporal_attention()
                if attention is not None:
                    batch_attention = attention.mean(dim=(0, 1))
                    attention_sum = (
                        batch_attention
                        if attention_sum is None
                        else attention_sum + batch_attention
                    )
                    attention_count += 1

        if feature_sum is not None and feature_count:
            feature_frames.append(
                pd.DataFrame(
                    {
                        "test_start": year,
                        "run_name": run_name,
                        "feature": _feature_names(deep_model.train_settings),
                        "mean_weight": (feature_sum / feature_count).numpy(),
                    }
                )
            )

        if attention_sum is not None and attention_count:
            attention_mean = (attention_sum / attention_count).numpy()
            attention_frame = pd.DataFrame(attention_mean)
            attention_frame.insert(0, "query_step", range(attention_mean.shape[0]))
            attention_frame.insert(0, "run_name", run_name)
            attention_frame.insert(0, "test_start", year)
            attention_frames.append(attention_frame)

    if not feature_frames:
        raise SystemExit("No interpretability batches were exported.")

    pd.concat(feature_frames, ignore_index=True).to_csv(
        output_dir / "feature_selection_weights.csv",
        index=False,
    )
    if attention_frames:
        pd.concat(attention_frames, ignore_index=True).to_csv(
            output_dir / "temporal_attention_mean.csv",
            index=False,
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest", default="bt-deepm-mt-vsn-k10-current")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "deepm_mt_vsn_20260625"),
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--seed-rank", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()

    out = export_interpretability(
        args.backtest,
        Path(args.output_dir),
        args.start_year,
        args.seed_rank,
        args.max_batches,
    )
    print(f"Wrote Momentum interpretability diagnostics to {out}")


if __name__ == "__main__":
    main()
