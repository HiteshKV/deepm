#!/usr/bin/env python
"""Benchmark DeRegiME training and validation throughput by device."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import gpytorch
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_deregime_device import (  # noqa: E402
    build_runtime,
    initialize_inducing_from_first_batch,
    load_experiment_config,
    one_training_step,
)
from deregime.training import (  # noqa: E402
    _build_pred_targets,
    _normalize_targets_with_current_revin,
    _prediction_likelihood_overrides,
    _uses_latent_mean_shift,
)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def validation_pass(config, model, likelihood, valid_loader, max_batches: int | None) -> dict:
    model.eval()
    likelihood.eval()
    mll_pred = gpytorch.mlls.VariationalELBO(
        likelihood,
        model.gp,
        num_data=max(1, len(valid_loader.dataset) * config.pred_len * model.D),
    )
    losses = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            seq_x, seq_y, seq_x_mark, seq_y_mark, _time_idx, _mh_y, _mh_m, _ = batch
            seq_x, seq_y, seq_x_mark, seq_y_mark = (
                t.to(config.device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
            )
            _mh_y = (
                _mh_y.to(config.device)
                if _mh_y is not None and hasattr(_mh_y, "to")
                else None
            )
            _mh_m = (
                _mh_m.to(config.device)
                if _mh_m is not None and hasattr(_mh_m, "to")
                else None
            )
            (
                feats_fit,
                feats_pred,
                _g_pred,
                aux,
                _g_pred_logits,
                _g_fit,
                batch_mean,
                _batch_std,
                deep_mean_fit,
                deep_mean_pred,
            ) = model.forward_features(
                seq_x,
                seq_x_mark,
                seq_y_mark,
                _mh_m,
                config.gp_label_len,
                config.pred_len,
            )
            y_pred_all = _build_pred_targets(seq_y, _mh_y, config.pred_len)
            y_target_norm = _normalize_targets_with_current_revin(
                model, y_pred_all, batch_mean
            )
            feats_all = torch.cat([feats_fit, feats_pred], dim=1)
            if _uses_latent_mean_shift(model):
                deep_mean_all = torch.cat([deep_mean_fit, deep_mean_pred], dim=1)
            else:
                deep_mean_all = None
            likelihood_overrides = _prediction_likelihood_overrides(aux)
            with gpytorch.settings.cholesky_jitter(1e-6):
                mvn_all = model.gp_mvn(feats_all)
                if deep_mean_all is not None:
                    mvn_all = gpytorch.distributions.MultivariateNormal(
                        mvn_all.mean + deep_mean_all, mvn_all.covariance_matrix
                    )
                feats_pred_only = feats_all[..., -config.pred_len :, :]
                raw_loss = -mll_pred(
                    mvn_all[..., -config.pred_len :],
                    y_target_norm,
                    feats_all=feats_pred_only,
                    kernel=model.gp.covar_module,
                    **likelihood_overrides,
                )
                if raw_loss.dim() > 0:
                    raw_loss = raw_loss.sum()
                losses.append(float(raw_loss.detach().cpu().item()))
    _sync(config.device)
    elapsed = time.perf_counter() - started
    return {
        "batches": len(losses),
        "elapsed_seconds": elapsed,
        "seconds_per_batch": elapsed / max(1, len(losses)),
        "mean_raw_loss": statistics.fmean(losses) if losses else None,
    }


def benchmark_device(args, device: str) -> dict:
    config = load_experiment_config(
        args.experiments_json, args.experiment, args.seed, device
    )
    model, likelihood, train_loader, valid_loader = build_runtime(config)
    initialize_inducing_from_first_batch(model, train_loader, config)
    model_params = list(model.parameters())
    seen_params = {id(param) for param in model_params}
    likelihood_params = [
        param for param in likelihood.parameters() if id(param) not in seen_params
    ]
    optimizer = torch.optim.Adam(model_params + likelihood_params, lr=float(config.lr))

    batch_times = []
    train_iter = iter(train_loader)
    for _ in range(args.num_batches):
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        result = one_training_step(
            config=config,
            model=model,
            likelihood=likelihood,
            train_loader=train_loader,
            batch=batch,
            optimizer=optimizer,
        )
        batch_times.append(result["elapsed_seconds"])

    validation = validation_pass(
        config,
        model,
        likelihood,
        valid_loader,
        None if args.max_valid_batches == 0 else args.max_valid_batches,
    )
    return {
        "status": "ok",
        "requested_device": device,
        "resolved_device": config.device,
        "train_batches": len(batch_times),
        "train_seconds_total": sum(batch_times),
        "train_seconds_per_batch_mean": statistics.fmean(batch_times)
        if batch_times
        else None,
        "train_seconds_per_batch_median": statistics.median(batch_times)
        if batch_times
        else None,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-json", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument(
        "--max-valid-batches",
        type=int,
        default=0,
        help="0 means full validation pass; positive values cap validation batches.",
    )
    args = parser.parse_args()

    results = []
    for device in args.devices:
        print(f"\nBenchmarking DeRegiME on {device}...")
        try:
            results.append(benchmark_device(args, device))
        except Exception as exc:
            results.append(
                {
                    "status": "failed",
                    "requested_device": device,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
