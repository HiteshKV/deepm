"""K-seed ensemble backtest for hourly equity DeePM pilot models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from equity_intraday_deepm.config import load_backtest_config, load_train_config
from equity_intraday_deepm.data import read_table
from equity_intraday_deepm.model import EquityHourlyModel
from equity_intraday_deepm.training import IntradayWindowDataset, resolve_device, _windows


def run_backtest(name: str, diagnostics: bool = False) -> Path:
    bt = load_backtest_config(name)
    train = load_train_config(bt["train_config"])
    device = resolve_device(bt.get("device", "auto"))
    frame = read_table(Path(bt["features_path"]))
    dataset = IntradayWindowDataset(frame, train["features"], int(train["active_top_n"]), int(train["seq_len"]))
    out_dir = Path(bt["output_dir"]) / bt.get("name", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = []
    for train_start, test_start, test_end in _windows(train, None)[::-1]:
        idx = np.where((dataset.years >= test_start) & (dataset.years < test_end))[0]
        if len(idx) == 0:
            continue
        positions.append(_predict_window(bt, train, dataset, idx, test_start, device))
    if not positions:
        raise RuntimeError("No backtest positions generated")
    pred = pd.concat(positions, ignore_index=True)
    returns, metrics = _returns_from_positions(pred, float(train["bars_per_year"]), int(train["active_top_n"]))
    pred.to_csv(out_dir / "positions.csv", index=False)
    returns.to_csv(out_dir / "returns.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    if diagnostics:
        _plot_nav(returns, out_dir / "nav_drawdown.png")
    print(f"backtest_dir={out_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return out_dir


def _predict_window(bt, train, dataset, indexes, test_start, device):
    root = Path(train["save_directory"]) / train["run_name"] / bt["architecture"] / str(test_start)
    best = pd.read_csv(root / "best_runs.csv", index_col=0).head(int(bt["top_n_seeds"]))
    x = dataset.x[indexes].to(device)
    preds = []
    for run_name in best.index.astype(str):
        settings = json.loads((root / "settings" / f"{run_name}.json").read_text(encoding="utf-8"))
        model = EquityHourlyModel(
            input_dim=len(train["features"]),
            active_top_n=int(train["active_top_n"]),
            hidden_dim=int(settings["hidden_dim"]),
            dropout=float(settings["dropout"]),
            num_heads=int(settings["num_heads"]),
            use_cross_attention=bool(train.get("use_cross_attention", True)),
        ).to(device)
        model.load_state_dict(torch.load(root / "models" / f"{run_name}.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            preds.append(model(x).detach().cpu().numpy())
    avg = np.mean(preds, axis=0)
    rows = []
    y = dataset.y[indexes].numpy()
    costs = dataset.cost[indexes].numpy()
    for local_i, sample_idx in enumerate(indexes):
        ts = dataset.timestamps[int(sample_idx)]
        for asset_i, sec in enumerate(dataset.security_ids[int(sample_idx)]):
            rows.append(
                {
                    "timestamp": ts,
                    "security_id": sec,
                    "position": float(avg[local_i, asset_i]),
                    "target": float(y[local_i, asset_i]),
                    "transaction_cost_bps": float(costs[local_i, asset_i]),
                    "window": int(test_start),
                }
            )
    return pd.DataFrame(rows)


def _returns_from_positions(pred: pd.DataFrame, bars_per_year: float, active_top_n: int):
    pred = pred.sort_values(["timestamp", "security_id"]).copy()
    wide = pred.pivot(index="timestamp", columns="security_id", values="position").fillna(0.0)
    targets = pred.pivot(index="timestamp", columns="security_id", values="target").reindex_like(wide).fillna(0.0)
    costs = pred.pivot(index="timestamp", columns="security_id", values="transaction_cost_bps").reindex_like(wide).ffill().fillna(0.0)
    turnover = wide.diff().abs().fillna(wide.abs())
    gross = (wide * targets).sum(axis=1) / active_top_n
    cost = (turnover * costs * 1e-4).sum(axis=1) / active_top_n
    net = gross - cost
    nav = (1.0 + net).cumprod()
    returns = pd.DataFrame({"timestamp": wide.index, "gross_return": gross.values, "cost": cost.values, "net_return": net.values, "nav": nav.values})
    metrics = _metrics(returns, bars_per_year)
    return returns, metrics


def _metrics(returns: pd.DataFrame, bars_per_year: float) -> dict[str, float | str | int]:
    net = returns["net_return"]
    gross = returns["gross_return"]
    nav = returns["nav"]
    drawdown = nav / nav.cummax() - 1.0
    def sharpe(series):
        std = series.std(ddof=0)
        return float(series.mean() / std * np.sqrt(bars_per_year)) if std else 0.0
    return {
        "start": str(pd.to_datetime(returns["timestamp"]).min()),
        "end": str(pd.to_datetime(returns["timestamp"]).max()),
        "bars": int(len(returns)),
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "total_return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "avg_cost": float(returns["cost"].mean()),
    }


def _plot_nav(returns: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ts = pd.to_datetime(returns["timestamp"])
    axes[0].plot(ts, returns["nav"])
    axes[0].set_title("Hourly top-10 ensemble NAV")
    dd = returns["nav"] / returns["nav"].cummax() - 1.0
    axes[1].fill_between(ts, dd, 0.0)
    axes[1].set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest hourly equity ensemble")
    parser.add_argument("--name", required=True)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args(argv)
    run_backtest(args.name, args.diagnostics)


if __name__ == "__main__":
    main()
