#!/usr/bin/env python
"""Benchmark DeePM training throughput across available torch devices."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import deepm.training.train as train_module  # noqa: E402
from deepm.configs.load import load_settings_for_architecture, load_sweep_settings  # noqa: E402
from deepm.data.dataset import unpack_torch_dataset  # noqa: E402
from deepm.models.base import DmnMode  # noqa: E402
from deepm.training.data_setup import (  # noqa: E402
    build_correlation_features,
    build_windows,
    compute_scalers,
    create_datasets,
    load_and_filter_data,
    validate_tickers_in_tcost,
)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _first_sweep_values(sweep_settings: dict) -> dict:
    values = {}
    for key, spec in sweep_settings.get("parameters", {}).items():
        if "values" in spec:
            values[key] = spec["values"][0]
        elif "value" in spec:
            values[key] = spec["value"]
    return values


def _prepare_window(settings: dict, filter_start_year: int | None):
    data = load_and_filter_data(settings, settings["data_parquet"], end_date=None)
    validate_tickers_in_tcost(settings, data)
    corr_feat = build_correlation_features(settings, data)
    windows = build_windows(settings, [filter_start_year] if filter_start_year else None)
    train_start, test_start, test_end = windows[-1]
    scalers = compute_scalers(data, settings, train_start, test_start)
    train_data, valid_data, test_data, train_extra_data, _ = create_datasets(
        settings,
        data,
        train_start,
        test_start,
        test_end,
        valid_end_optional=None,
        corr_feat=corr_feat,
    )
    return test_start, test_end, {**settings, **scalers}, train_data, valid_data, test_data, train_extra_data


def _benchmark_device(
    requested_device: str,
    architecture: str,
    settings: dict,
    hp: dict,
    train_data,
    valid_data,
    test_data,
    train_extra_data,
    warmup_batches: int,
    measure_batches: int,
) -> dict:
    try:
        resolved = train_module.set_training_device(requested_device)
        with tempfile.TemporaryDirectory(prefix="deepm-device-bench-") as tmp_dir:
            trainer = train_module.TrainDeepMomentumNetwork(
                train_data,
                valid_data,
                test_data,
                settings["seq_len"],
                len(settings["features"]),
                save_path=tmp_dir,
                train_extra_data=train_extra_data,
            )
            model_kwargs = {**settings, **hp}
            model_kwargs.pop("optimise_loss_function", None)
            model = trainer._load_architecture(
                architecture=architecture,
                input_dim=len(settings["features"]),
                num_tickers=valid_data.num_tickers,
                optimise_loss_function=settings["optimise_loss_function"],
                **model_kwargs,
            )
            model.train()
            optimizer_kwargs = dict(model_kwargs)
            optimizer_kwargs.pop("lr", None)
            optimizer_kwargs.pop("weight_decay", None)
            optimizer = trainer._build_optimizer(
                model,
                hp["lr"],
                hp.get("weight_decay", 0.0),
                **optimizer_kwargs,
            )
            batch_size = int(hp["batch_size"])
            loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            samples_list = list(loader)
            if not samples_list:
                raise RuntimeError("Training dataset produced no benchmark batches.")

            timings: list[float] = []
            total_needed = warmup_batches + measure_batches
            for batch_idx in range(total_needed):
                samples = samples_list[batch_idx % len(samples_list)]
                _sync(resolved)
                start = time.perf_counter()
                loss = model(
                    **unpack_torch_dataset(
                        samples,
                        train_data,
                        resolved,
                        use_dates_mask=False,
                        live_mode=False,
                    ),
                    mode=DmnMode.TRAINING,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hp["max_gradient_norm"]
                )
                optimizer.step()
                _sync(resolved)
                elapsed = time.perf_counter() - start
                if batch_idx >= warmup_batches:
                    timings.append(elapsed)

            if not timings:
                raise RuntimeError("No benchmark batches were measured.")
            return {
                "requested_device": requested_device,
                "resolved_device": str(resolved),
                "status": "ok",
                "batches": len(timings),
                "mean_seconds_per_batch": statistics.mean(timings),
                "median_seconds_per_batch": statistics.median(timings),
                "min_seconds_per_batch": min(timings),
                "max_seconds_per_batch": max(timings),
            }
    except Exception as exc:  # pragma: no cover - exercised by local hardware
        return {
            "requested_device": requested_device,
            "resolved_device": requested_device,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--run", default="deepm-mt-vsn")
    parser.add_argument("-a", "--arch", default="MT_DEEPM")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to test. Defaults to all locally available devices.",
    )
    parser.add_argument(
        "-fsy",
        "--filter-start-year",
        type=int,
        default=2020,
        help="Rolling test start year used to build the benchmark dataset.",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=2)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings_for_architecture(args.run, args.arch)
    sweep_settings = load_sweep_settings(settings["sweep_yaml"])
    hp = _first_sweep_values(sweep_settings)
    benchmark_settings = copy.deepcopy(settings)

    test_start, test_end, benchmark_settings, train_data, valid_data, test_data, train_extra_data = (
        _prepare_window(benchmark_settings, args.filter_start_year)
    )
    devices = args.devices or train_module.available_training_devices()
    results = [
        _benchmark_device(
            device,
            args.arch,
            benchmark_settings,
            hp,
            train_data,
            valid_data,
            test_data,
            train_extra_data,
            args.warmup_batches,
            args.measure_batches,
        )
        for device in devices
    ]
    successful = [row for row in results if row["status"] == "ok"]
    recommended = (
        min(successful, key=lambda row: row["mean_seconds_per_batch"])["resolved_device"]
        if successful
        else None
    )
    payload = {
        "run": args.run,
        "architecture": args.arch,
        "test_window": f"{test_start}-{test_end}",
        "hyperparameters": hp,
        "results": results,
        "recommended_device": recommended,
    }

    print(json.dumps(payload, indent=2))
    if recommended:
        print(f"\nRecommended command override: --device {recommended}")
        print(f"Recommended env override: export DEEPM_TRAIN_DEVICE={recommended}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
