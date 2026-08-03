#!/usr/bin/env python
"""Run one DeRegiME training-step canary on a requested torch device."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gpytorch
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deregime.config import CONFIGS, ConfigDict  # noqa: E402
from deregime.data import load_long_to_wide, make_splits_and_loaders  # noqa: E402
from deregime.run import (  # noqa: E402
    apply_overrides,
    build_regime_model_and_likelihood,
    configure_runtime_device,
)
from deregime.training import (  # noqa: E402
    _build_pred_targets,
    _normalize_targets_with_current_revin,
    _prediction_likelihood_overrides,
    _uses_latent_mean_shift,
)


def load_experiment_config(path: Path, experiment: str, seed: int, device: str) -> ConfigDict:
    experiments = json.loads(path.read_text(encoding="utf-8"))
    match = next((item for item in experiments if item.get("name") == experiment), None)
    if match is None:
        names = ", ".join(item.get("name", "<unnamed>") for item in experiments)
        raise SystemExit(f"Experiment '{experiment}' not found in {path}. Found: {names}")
    cfg = ConfigDict(
        apply_overrides(CONFIGS, {**match.get("overrides", {}), "seed": seed, "device": device})
    )
    configure_runtime_device(cfg)
    return cfg


def build_runtime(config: ConfigDict):
    wide = load_long_to_wide(
        config.file, config.date_col, config.series_id_col, config.value_col
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _targ_scaler,
        _train_dataset,
        train_loader,
        _,
        valid_loader,
        _,
        _test_loader,
        tmark_dim,
        series_dim,
        target_dim,
    ) = make_splits_and_loaders(wide, config)
    model, likelihood = build_regime_model_and_likelihood(
        config, series_dim, target_dim, tmark_dim
    )
    return model, likelihood, train_loader, valid_loader


def initialize_inducing_from_first_batch(model, train_loader, config: ConfigDict) -> None:
    batch = next(iter(train_loader))
    seq_x, _, seq_x_mark, seq_y_mark, _, _, _mh_m, _ = batch
    model.initialize_inducing_from_batch(
        seq_x.to(config.device),
        seq_x_mark.to(config.device),
        seq_y_mark.to(config.device),
        _mh_m.to(config.device) if _mh_m is not None else None,
        config.gp_label_len,
        config.pred_len,
        config.num_inducing_points,
    )


def one_training_step(
    *,
    config: ConfigDict,
    model,
    likelihood,
    train_loader,
    batch=None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    if batch is None:
        batch = next(iter(train_loader))
    if len(batch) != 8:
        raise NotImplementedError("Multi-horizon inputs required for DeRegiME canary.")

    seq_x, seq_y, seq_x_mark, seq_y_mark, _time_idx, _mh_y, _mh_m, _ = batch
    seq_x, seq_y, seq_x_mark, seq_y_mark = (
        t.to(config.device) for t in [seq_x, seq_y, seq_x_mark, seq_y_mark]
    )
    _mh_y = _mh_y.to(config.device) if _mh_y is not None and hasattr(_mh_y, "to") else None
    _mh_m = _mh_m.to(config.device) if _mh_m is not None and hasattr(_mh_m, "to") else None

    model.train()
    likelihood.train()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)
        likelihood.zero_grad(set_to_none=True)

    started = time.perf_counter()
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
    y_target_norm = _normalize_targets_with_current_revin(model, y_pred_all, batch_mean)
    feats_all = torch.cat([feats_fit, feats_pred], dim=1)
    deep_mean_all = (
        torch.cat([deep_mean_fit, deep_mean_pred], dim=1)
        if _uses_latent_mean_shift(model)
        else None
    )
    likelihood_overrides = _prediction_likelihood_overrides(aux)
    mll_pred = gpytorch.mlls.VariationalELBO(
        likelihood,
        model.gp,
        num_data=max(1, len(train_loader.dataset) * config.pred_len * model.D),
    )

    with gpytorch.settings.cholesky_jitter(1e-6):
        mvn_all = model.gp_mvn(feats_all)
        if deep_mean_all is not None:
            mvn_all = gpytorch.distributions.MultivariateNormal(
                mvn_all.mean + deep_mean_all, mvn_all.covariance_matrix
            )
        feats_pred_only = feats_all[..., -config.pred_len :, :]
        raw_elbo = -mll_pred(
            mvn_all[..., -config.pred_len :],
            y_target_norm,
            feats_all=feats_pred_only,
            kernel=model.gp.covar_module,
            **likelihood_overrides,
        )
        if raw_elbo.dim() > 0:
            raw_elbo = raw_elbo.sum()
        loss = raw_elbo / max(1, len(train_loader.dataset) * config.pred_len * model.D)

    loss.backward()
    if optimizer is not None:
        optimizer.step()
    if config.device == "mps":
        torch.mps.synchronize()
    elif str(config.device).startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    return {"loss": float(loss.detach().cpu().item()), "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-json", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    config = load_experiment_config(
        args.experiments_json, args.experiment, args.seed, args.device
    )
    print(
        f"DeRegiME device canary: requested={config.requested_device} "
        f"resolved={config.device}"
    )
    try:
        model, likelihood, train_loader, _valid_loader = build_runtime(config)
        initialize_inducing_from_first_batch(model, train_loader, config)
        result = one_training_step(
            config=config, model=model, likelihood=likelihood, train_loader=train_loader
        )
    except Exception as exc:
        print(
            f"DeRegiME device canary failed on {config.device}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "If this is an MPS run, use CPU mode or reduce the model until the "
            "unsupported operation is isolated.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(json.dumps({"status": "ok", "device": config.device, **result}, indent=2))


if __name__ == "__main__":
    main()
