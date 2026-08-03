"""Bridge utilities for DeRegiME sidecar features in DeePM.

The bridge keeps DeePM's training objective unchanged. It builds optional
``dg_`` feature columns that can be merged into a DeePM feature cache and used
by ablation configs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm


DEFAULT_HORIZONS = (1, 5, 21, 63)
DEFAULT_NUM_REGIMES = 4
DEFAULT_SEQ_LEN = 252
DEFAULT_MIN_HISTORY = 126
DEFAULT_TARGET_LAG = 1


def normal_cdf(x: pd.Series) -> pd.Series:
    """Vectorized standard normal CDF without requiring scipy arrays upstream."""
    return pd.Series(
        0.5 * (1.0 + erf(np.asarray(x, dtype=float) / np.sqrt(2.0))),
        index=x.index,
    )


def regime_feature_columns(
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    num_regimes: int = DEFAULT_NUM_REGIMES,
) -> list[str]:
    """Return all feature columns emitted by the sidecar generator."""
    cols = ["dg_available"]
    for h in horizons:
        cols.extend(
            [
                f"dg_mu_h{h}",
                f"dg_sigma_h{h}",
                f"dg_p_pos_h{h}",
                f"dg_var05_h{h}",
                f"dg_es05_h{h}",
                f"dg_regime_entropy_h{h}",
                f"dg_regime_maxprob_h{h}",
                f"dg_regime_id_h{h}",
            ]
        )
        cols.extend(f"dg_pi{r}_h{h}" for r in range(num_regimes))
    return cols


def feature_group_columns(
    group: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    num_regimes: int = DEFAULT_NUM_REGIMES,
) -> list[str]:
    """Return the DeRegiME columns used by a DeePM ablation group."""
    horizons = tuple(horizons)
    if group not in {"mean", "risk", "full"}:
        raise ValueError("group must be one of: mean, risk, full")

    cols = ["dg_available"]
    for h in horizons:
        cols.extend([f"dg_mu_h{h}", f"dg_p_pos_h{h}"])
        if group in {"mean", "full"}:
            cols.extend(f"dg_pi{r}_h{h}" for r in range(num_regimes))
        if group in {"risk", "full"}:
            cols.extend(
                [
                    f"dg_sigma_h{h}",
                    f"dg_var05_h{h}",
                    f"dg_es05_h{h}",
                    f"dg_regime_entropy_h{h}",
                ]
            )
        if group == "full":
            cols.extend(
                [
                    f"dg_regime_maxprob_h{h}",
                    f"dg_regime_id_h{h}",
                ]
            )
    return cols


def filter_deepm_features(features: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Apply the same ticker subset used by a DeePM training config."""
    data = features.copy()
    if settings.get("ticker_subset"):
        data = data[data["ticker"].isin(settings["ticker_subset"])]
    return data


def write_deregime_long_csv(
    features: pd.DataFrame,
    output: Path,
    *,
    target_col: str = "target",
) -> Path:
    """Write a DeRegiME-compatible long CSV: ``date,cols,data``."""
    if target_col not in features.columns:
        raise ValueError(f"target column not found: {target_col}")

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.index)
    frame = frame.reset_index(drop=True)
    long = (
        frame[["date", "ticker", target_col]]
        .rename(columns={"ticker": "cols", target_col: "data"})
        .dropna(subset=["data"])
        .sort_values(["date", "cols"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output, index=False)
    return output


def _rolling_regime_probabilities(
    hist: pd.Series,
    *,
    seq_len: int,
    min_history: int,
    num_regimes: int,
) -> pd.DataFrame:
    """Causal regime probabilities from lagged standardized returns."""
    mean = hist.ewm(span=seq_len, min_periods=min_history, adjust=False).mean()
    std = hist.ewm(span=seq_len, min_periods=min_history, adjust=False).std()
    z = (hist - mean) / std.replace(0.0, np.nan)

    if num_regimes != 4:
        raise ValueError("The v1 DeePM bridge reports exactly four regimes.")

    buckets = [
        z < -1.0,
        (z >= -1.0) & (z < 0.0),
        (z >= 0.0) & (z < 1.0),
        z >= 1.0,
    ]
    probs = pd.concat(
        [
            bucket.astype(float)
            .ewm(span=seq_len, min_periods=min_history, adjust=False)
            .mean()
            for bucket in buckets
        ],
        axis=1,
    )
    probs.columns = [f"pi{r}" for r in range(num_regimes)]
    row_sum = probs.sum(axis=1).replace(0.0, np.nan)
    probs = probs.div(row_sum, axis=0)
    return probs


def _entropy(probs: pd.DataFrame) -> pd.Series:
    clipped = probs.clip(lower=1e-12)
    return -(clipped * np.log(clipped)).sum(axis=1) / np.log(probs.shape[1])


def build_causal_proxy_sidecar(
    features: pd.DataFrame,
    *,
    target_col: str = "target",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    seq_len: int = DEFAULT_SEQ_LEN,
    min_history: int = DEFAULT_MIN_HISTORY,
    target_lag: int = DEFAULT_TARGET_LAG,
    num_regimes: int = DEFAULT_NUM_REGIMES,
) -> pd.DataFrame:
    """Build causal DeRegiME-style probabilistic features.

    The proxy is intentionally simple and deterministic. It gives the DeePM
    ablations the same sidecar interface that real DeRegiME artifact exports
    use, while tests and cache validation stay fast.
    """
    if target_col not in features.columns:
        raise ValueError(f"target column not found: {target_col}")
    if target_lag < 1:
        raise ValueError("target_lag must be at least 1 to avoid target leakage")

    frames = []
    for ticker, group in features.sort_index().groupby("ticker", sort=True):
        group = group.sort_index()
        hist = group[target_col].astype(float).shift(target_lag)
        available = hist.expanding(min_periods=min_history).count() >= min_history
        out = pd.DataFrame(index=group.index)
        out["ticker"] = ticker
        out["dg_available"] = available.astype(float)

        base_probs = _rolling_regime_probabilities(
            hist,
            seq_len=seq_len,
            min_history=min_history,
            num_regimes=num_regimes,
        )

        for h in horizons:
            span = max(min_history, int(round(seq_len / np.sqrt(float(h)))))
            mu = hist.ewm(span=span, min_periods=min_history, adjust=False).mean()
            sigma = (
                hist.ewm(span=span, min_periods=min_history, adjust=False)
                .std()
                .abs()
                * np.sqrt(float(h))
            )
            sigma = sigma.replace(0.0, np.nan)
            z = mu / sigma
            p_pos = normal_cdf(z)
            var05 = mu + norm.ppf(0.05) * sigma
            es05 = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05

            probs = base_probs.copy()
            probs.columns = [f"dg_pi{r}_h{h}" for r in range(num_regimes)]

            out[f"dg_mu_h{h}"] = mu
            out[f"dg_sigma_h{h}"] = sigma
            out[f"dg_p_pos_h{h}"] = p_pos
            out[f"dg_var05_h{h}"] = var05
            out[f"dg_es05_h{h}"] = es05
            out[f"dg_regime_entropy_h{h}"] = _entropy(probs)
            out[f"dg_regime_maxprob_h{h}"] = probs.max(axis=1)
            out[f"dg_regime_id_h{h}"] = probs.to_numpy().argmax(axis=1)
            out = out.join(probs)

        dg_cols = [c for c in out.columns if c.startswith("dg_")]
        out.loc[~available, dg_cols] = np.nan
        out[dg_cols] = out[dg_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frames.append(out)

    sidecar = pd.concat(frames).sort_values(["ticker"], kind="stable").sort_index()
    sidecar.index = pd.to_datetime(sidecar.index)
    return sidecar


def compile_deregime_artifact_sidecar(
    predictions_dir: Path,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    num_regimes: int = DEFAULT_NUM_REGIMES,
    scaled: bool = True,
) -> pd.DataFrame:
    """Compile DeRegiME prediction/activation CSV artifacts into ``dg_`` columns."""
    frames = []
    suffix = "scaled__" if scaled else ""
    for h in horizons:
        pred_path = predictions_dir / f"predictions_{suffix}h{h}.csv"
        if not pred_path.exists():
            pred_path = predictions_dir / f"predictions__h{h}.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing DeRegiME predictions for h={h}: {pred_path}")

        pred = pd.read_csv(pred_path, parse_dates=["time"])
        pred = pred.rename(columns={"time": "date", "channel_name": "ticker"})
        sigma = pred["y_std"].abs().replace(0.0, np.nan)
        z = pred["y_mean"] / sigma
        frame = pred[["date", "ticker"]].copy()
        frame[f"dg_mu_h{h}"] = pred["y_mean"]
        frame[f"dg_sigma_h{h}"] = sigma
        frame[f"dg_p_pos_h{h}"] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        frame[f"dg_var05_h{h}"] = pred["y_mean"] + norm.ppf(0.05) * sigma
        frame[f"dg_es05_h{h}"] = pred["y_mean"] - sigma * norm.pdf(norm.ppf(0.05)) / 0.05

        act_path = predictions_dir / f"activations__h{h}.csv"
        if act_path.exists():
            acts = pd.read_csv(act_path, parse_dates=["time"])
            acts = acts.rename(columns={"time": "date", "channel_name": "ticker"})
            r_cols = [c for c in acts.columns if c.startswith("R")]
            acts = acts[["date", "ticker", *r_cols]].copy()
            for r in range(num_regimes):
                src = f"R{r + 1}"
                acts[f"dg_pi{r}_h{h}"] = acts[src] if src in acts else 0.0
            pi_cols = [f"dg_pi{r}_h{h}" for r in range(num_regimes)]
            acts = acts[["date", "ticker", *pi_cols]]
        else:
            acts = frame[["date", "ticker"]].copy()
            for r in range(num_regimes):
                acts[f"dg_pi{r}_h{h}"] = 1.0 / num_regimes

        frame = frame.merge(acts, on=["date", "ticker"], how="left")
        pi_cols = [f"dg_pi{r}_h{h}" for r in range(num_regimes)]
        probs = frame[pi_cols].fillna(1.0 / num_regimes)
        row_sum = probs.sum(axis=1).replace(0.0, np.nan)
        probs = probs.div(row_sum, axis=0).fillna(1.0 / num_regimes)
        frame[pi_cols] = probs
        frame[f"dg_regime_entropy_h{h}"] = _entropy(probs)
        frame[f"dg_regime_maxprob_h{h}"] = probs.max(axis=1)
        frame[f"dg_regime_id_h{h}"] = probs.to_numpy().argmax(axis=1)
        frames.append(frame)

    sidecar = frames[0]
    for frame in frames[1:]:
        sidecar = sidecar.merge(frame, on=["date", "ticker"], how="outer")
    sidecar["dg_available"] = 1.0
    sidecar = sidecar.set_index("date").sort_index()
    dg_cols = [c for c in sidecar.columns if c.startswith("dg_")]
    sidecar[dg_cols] = sidecar[dg_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sidecar


def merge_sidecar_features(
    base_features: pd.DataFrame,
    sidecar: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    num_regimes: int = DEFAULT_NUM_REGIMES,
) -> pd.DataFrame:
    """Merge sidecar columns into a DeePM feature cache."""
    dg_cols = regime_feature_columns(horizons, num_regimes)

    base = base_features.copy()
    base["_row_order"] = np.arange(len(base))
    base["date"] = pd.to_datetime(base.index)
    base = base.reset_index(drop=True)

    dg = sidecar.copy()
    dg["date"] = pd.to_datetime(dg.index)
    dg = dg.reset_index(drop=True)
    missing = set(dg_cols) - set(dg.columns)
    if missing:
        raise ValueError(f"sidecar is missing dg columns: {sorted(missing)}")

    duplicate_mask = dg.duplicated(["date", "ticker"], keep=False)
    if duplicate_mask.any():
        dupes = dg.loc[duplicate_mask, ["date", "ticker"]].head(10).to_dict("records")
        raise ValueError(f"sidecar has duplicate date/ticker rows: {dupes}")

    merged = base.merge(dg[["date", "ticker", *dg_cols]], on=["date", "ticker"], how="left")
    merged[dg_cols] = merged[dg_cols].fillna(0.0)
    merged = merged.sort_values("_row_order").drop(columns=["_row_order", "date"])
    merged.index = base_features.index
    return merged


def validate_sidecar_alignment(
    base_features: pd.DataFrame,
    sidecar: pd.DataFrame,
    *,
    expected_tickers: int | None = None,
) -> dict:
    """Return validation metadata and raise on structural alignment failures."""
    base_pairs = (
        base_features.assign(date=pd.to_datetime(base_features.index))
        .reset_index(drop=True)[["date", "ticker"]]
        .drop_duplicates()
    )
    side_pairs = (
        sidecar.assign(date=pd.to_datetime(sidecar.index))
        .reset_index(drop=True)[["date", "ticker"]]
        .drop_duplicates()
    )
    if len(side_pairs) != len(sidecar):
        raise ValueError("sidecar contains duplicate date/ticker rows")

    missing = base_pairs.merge(side_pairs, on=["date", "ticker"], how="left", indicator=True)
    missing_count = int((missing["_merge"] == "left_only").sum())
    ticker_count = int(sidecar["ticker"].nunique())
    if expected_tickers is not None and ticker_count != expected_tickers:
        raise ValueError(f"expected {expected_tickers} sidecar tickers, found {ticker_count}")

    return {
        "base_rows": int(len(base_features)),
        "sidecar_rows": int(len(sidecar)),
        "missing_base_pairs": missing_count,
        "sidecar_tickers": ticker_count,
        "sidecar_first_date": str(pd.to_datetime(sidecar.index).min().date()),
        "sidecar_last_date": str(pd.to_datetime(sidecar.index).max().date()),
    }


def write_sidecar_audit(
    path: Path,
    *,
    method: str,
    settings: dict,
    input_features: Path,
    sidecar_output: Path,
    merged_output: Path,
    target_col: str,
    horizons: Iterable[int],
    seq_len: int,
    min_history: int,
    target_lag: int,
    validation: dict,
    leakage_policy: str | None = None,
) -> Path:
    """Write immutable metadata for leakage review and experiment tracking."""
    test_starts = list(settings.get("test_start_years", []))
    final_test_year = settings.get("final_test_year")
    windows = []
    for i, start in enumerate(test_starts):
        end = test_starts[i + 1] if i + 1 < len(test_starts) else final_test_year + 1
        windows.append(
            {
                "train_start_year": settings.get("first_train_year"),
                "test_start_year": start,
                "test_end_year_exclusive": end,
                "sidecar_training_cutoff_policy": (
                    "rolling_proxy_features use only target values shifted by "
                    f"{target_lag} row(s) within each ticker"
                ),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "input_features": str(input_features),
        "sidecar_output": str(sidecar_output),
        "merged_output": str(merged_output),
        "target_col": target_col,
        "horizons": list(horizons),
        "seq_len": seq_len,
        "min_history": min_history,
        "target_lag": target_lag,
        "feature_columns": regime_feature_columns(horizons),
        "validation": validation,
        "windows": windows,
        "leakage_policy": leakage_policy or (
            "No same-row DeePM target is used as a feature. Proxy features at "
            "date t are calculated from target history no later than t-1."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
