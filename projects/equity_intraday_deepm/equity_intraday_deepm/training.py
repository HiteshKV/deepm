"""Training CLI for hourly top-N equity DeePM pilot models."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from equity_intraday_deepm.config import load_train_config, project_path
from equity_intraday_deepm.data import read_table
from equity_intraday_deepm.model import EquityHourlyModel, sharpe_score, softmin_sharpe_loss


class IntradayWindowDataset(Dataset):
    """Tensorized dynamic top-N hourly dataset."""

    def __init__(self, frame: pd.DataFrame, features: list[str], active_top_n: int, seq_len: int) -> None:
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["security_id"] = data["security_id"].astype(str)
        self.features = features
        self.active_top_n = active_top_n
        self.seq_len = seq_len
        self.by_security = {
            sec: g.sort_values("timestamp").set_index("timestamp")
            for sec, g in data.groupby("security_id", sort=False)
        }
        active = data[data["is_active"].astype(bool)].copy()
        self.timestamps = []
        self.security_ids = []
        self.samples_x = []
        self.samples_y = []
        self.samples_cost = []
        for ts, group in active.groupby("timestamp", sort=True):
            group = group.sort_values("rank")
            if len(group) != active_top_n:
                continue
            x_rows, y_rows, cost_rows, ids = [], [], [], []
            ok = True
            for row in group.itertuples(index=False):
                sec = str(row.security_id)
                hist = self.by_security[sec].loc[:ts].tail(seq_len)
                if len(hist) < seq_len:
                    ok = False
                    break
                values = hist[features].astype(float).values
                target = float(getattr(row, "target"))
                cost = float(getattr(row, "transaction_cost_bps"))
                if not np.isfinite(values).all() or not np.isfinite(target):
                    ok = False
                    break
                x_rows.append(values)
                y_rows.append(target)
                cost_rows.append(cost)
                ids.append(sec)
            if ok:
                self.timestamps.append(pd.Timestamp(ts))
                self.security_ids.append(ids)
                self.samples_x.append(np.asarray(x_rows, dtype=np.float32))
                self.samples_y.append(np.asarray(y_rows, dtype=np.float32))
                self.samples_cost.append(np.asarray(cost_rows, dtype=np.float32))
        if not self.samples_x:
            raise ValueError("No valid hourly top-N sequence samples could be built")
        self.x = torch.tensor(np.asarray(self.samples_x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.samples_y), dtype=torch.float32)
        self.cost = torch.tensor(np.asarray(self.samples_cost), dtype=torch.float32)
        self.years = np.asarray([ts.year for ts in self.timestamps])

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.cost[idx]


def resolve_device(requested: str) -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device request: {requested}")


def train_from_config(run_name: str, arch: str, device_request: str | None = None, filter_start_years: list[int] | None = None) -> None:
    cfg = load_train_config(run_name)
    device = resolve_device(device_request or cfg.get("device", "auto"))
    data = read_table(Path(cfg["features_path"]))
    dataset = IntradayWindowDataset(data, cfg["features"], int(cfg["active_top_n"]), int(cfg["seq_len"]))
    windows = _windows(cfg, filter_start_years)
    for train_start, test_start, test_end in windows:
        _train_window(cfg, arch, dataset, train_start, test_start, test_end, device)


def _train_window(cfg: dict[str, Any], arch: str, dataset: IntradayWindowDataset, train_start: int, test_start: int, test_end: int, device: torch.device) -> None:
    years = dataset.years
    train_valid_idx = np.where((years >= train_start) & (years < test_start))[0]
    test_idx = np.where((years >= test_start) & (years < test_end))[0]
    if len(train_valid_idx) < 4 or len(test_idx) < 1:
        raise ValueError(f"Insufficient samples for window {test_start}: train={len(train_valid_idx)} test={len(test_idx)}")
    cutoff = max(1, int(len(train_valid_idx) * float(cfg["train_valid_split"])))
    train_idx, valid_idx = train_valid_idx[:cutoff], train_valid_idx[cutoff:]
    if len(valid_idx) == 0:
        valid_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    save_dir = Path(cfg["save_directory"]) / cfg["run_name"] / arch / str(test_start)
    for sub in ["models", "settings", "data-params"]:
        (save_dir / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    existing = _read_all_runs(save_dir)
    for i in range(int(cfg["random_search_max_iterations"])):
        run_id = f"seed-{i:03d}"
        if run_id in existing.index and (save_dir / "models" / f"{run_id}.pt").exists():
            cached = existing.loc[run_id].to_dict()
            cached["run_name"] = run_id
            rows.append(cached)
            continue
        hparams = _sample_hparams(cfg, i)
        metrics = _train_seed(cfg, hparams, dataset, train_idx, valid_idx, test_idx, device, save_dir, run_id)
        rows.append(metrics)
        _write_runs(save_dir, rows)
    _write_runs(save_dir, rows)


def _train_seed(cfg, hparams, dataset, train_idx, valid_idx, test_idx, device, save_dir, run_id):
    torch.manual_seed(int(hparams["seed"]))
    random.seed(int(hparams["seed"]))
    np.random.seed(int(hparams["seed"]))
    model = EquityHourlyModel(
        input_dim=len(cfg["features"]),
        active_top_n=int(cfg["active_top_n"]),
        hidden_dim=int(hparams["hidden_dim"]),
        dropout=float(hparams["dropout"]),
        num_heads=int(hparams["num_heads"]),
        use_cross_attention=bool(cfg.get("use_cross_attention", True)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)
    best_valid = -float("inf")
    patience = 0
    best_path = save_dir / "models" / f"{run_id}.pt"
    for epoch in range(int(cfg["iterations"])):
        model.train()
        for x, y, cost in train_loader:
            x, y, cost = x.to(device), y.to(device), cost.to(device)
            pos = model(x)
            returns = _portfolio_returns(pos, y, cost, cfg)
            loss = _objective_loss(returns, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_gradient_norm", 1.0)))
            opt.step()
        valid = _eval_model(model, dataset, valid_idx, cfg, device)
        if epoch + 1 <= int(cfg["val_burnin_steps"]):
            continue
        if valid >= best_valid + float(cfg["min_delta"]):
            best_valid = valid
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= int(cfg["early_stopping"]):
                break
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
        best_valid = _eval_model(model, dataset, valid_idx, cfg, device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_sharpe = _eval_model(model, dataset, test_idx, cfg, device)
    settings = {**cfg, **hparams, "input_dim": len(cfg["features"])}
    (save_dir / "settings" / f"{run_id}.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    (save_dir / "data-params" / f"{run_id}.json").write_text(
        json.dumps({"features": cfg["features"], "active_top_n": cfg["active_top_n"], "seq_len": cfg["seq_len"]}, indent=2),
        encoding="utf-8",
    )
    return {
        "run_name": run_id,
        "valid_loss_best": float(best_valid),
        "test_sharpe_net": float(test_sharpe),
        "test_sharpe_gross": float(test_sharpe),
        **hparams,
    }


def _portfolio_returns(pos: torch.Tensor, y: torch.Tensor, cost: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    gross = (pos * y).mean(dim=1)
    cost_return = (pos.abs() * cost * 1e-4).mean(dim=1) * float(cfg.get("cost_penalty", 1.0))
    return gross - cost_return


def _objective_loss(returns: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    if cfg.get("use_softmin", True):
        return softmin_sharpe_loss(returns, float(cfg["bars_per_year"]), beta=float(cfg.get("softmin_beta", 5.0)))
    return -sharpe_score(returns, float(cfg["bars_per_year"]))


def _eval_model(model, dataset, indexes, cfg, device) -> float:
    loader = DataLoader(Subset(dataset, indexes), batch_size=int(cfg["batch_size"]), shuffle=False)
    returns = []
    model.eval()
    with torch.no_grad():
        for x, y, cost in loader:
            ret = _portfolio_returns(model(x.to(device)), y.to(device), cost.to(device), cfg)
            returns.append(ret.detach().cpu())
    if not returns:
        return float("nan")
    return float(sharpe_score(torch.cat(returns), float(cfg["bars_per_year"])).item())


def _sample_hparams(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    search = cfg.get("search_space", {})
    def pick(name, default):
        values = search.get(name, [default])
        return values[rng.randrange(len(values))]
    return {
        "seed": seed,
        "hidden_dim": int(pick("hidden_dim", cfg.get("hidden_dim", 64))),
        "dropout": float(pick("dropout", cfg.get("dropout", 0.1))),
        "lr": float(pick("lr", cfg.get("lr", 1e-3))),
        "weight_decay": float(pick("weight_decay", cfg.get("weight_decay", 1e-4))),
        "num_heads": int(pick("num_heads", cfg.get("num_heads", 4))),
    }


def _windows(cfg: dict[str, Any], filter_start_years: list[int] | None) -> list[tuple[int, int, int]]:
    starts = list(cfg["test_start_years"])
    ends = starts[1:] + [int(cfg["final_test_year"]) + 1]
    windows = [(int(cfg["first_train_year"]), int(s), int(e)) for s, e in zip(starts, ends)]
    if filter_start_years:
        windows = [w for w in windows if w[1] in set(filter_start_years)]
    return windows[::-1]


def _read_all_runs(save_dir: Path) -> pd.DataFrame:
    path = save_dir / "all_runs.csv"
    if not path.exists():
        return pd.DataFrame().set_index(pd.Index([], name="run_name"))
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    return frame


def _write_runs(save_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).drop_duplicates("run_name", keep="last").set_index("run_name")
    frame = frame.sort_values("valid_loss_best", ascending=False)
    frame.to_csv(save_dir / "all_runs.csv")
    frame.head(10).to_csv(save_dir / "best_runs.csv")
    (save_dir / "best_runs_mean.json").write_text(
        json.dumps(frame.head(10)[["valid_loss_best", "test_sharpe_net"]].mean().to_dict(), indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train hourly equity DeePM pilot")
    parser.add_argument("-r", "--run", required=True)
    parser.add_argument("-a", "--arch", required=True, choices=["EQH_MT_VSN", "EQH_DEEPM_XATT"])
    parser.add_argument("--device", default=None)
    parser.add_argument("-fsy", "--filter-start-years", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    train_from_config(args.run, args.arch, args.device, args.filter_start_years)


if __name__ == "__main__":
    main()
