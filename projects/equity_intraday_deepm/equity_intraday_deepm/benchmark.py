"""Device benchmark for the hourly equity model."""

from __future__ import annotations

import argparse
import json
import time

import torch

from equity_intraday_deepm.config import load_train_config
from equity_intraday_deepm.model import EquityHourlyModel
from equity_intraday_deepm.training import resolve_device


def benchmark(run: str, devices: list[str]) -> dict[str, object]:
    cfg = load_train_config(run)
    results = {}
    for name in devices:
        try:
            device = resolve_device(name)
            model = EquityHourlyModel(
                input_dim=len(cfg["features"]),
                active_top_n=int(cfg["active_top_n"]),
                hidden_dim=int(cfg.get("hidden_dim", 64)),
                num_heads=int(cfg.get("num_heads", 4)),
                use_cross_attention=bool(cfg.get("use_cross_attention", True)),
            ).to(device)
            x = torch.randn(8, int(cfg["active_top_n"]), int(cfg["seq_len"]), len(cfg["features"]), device=device)
            y = torch.randn(8, int(cfg["active_top_n"]), device=device) * 0.001
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            torch.manual_seed(7)
            start = time.perf_counter()
            for _ in range(5):
                loss = -(model(x) * y).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            results[name] = {"seconds_per_step": (time.perf_counter() - start) / 5.0, "available": True}
        except Exception as exc:
            results[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    available = {k: v["seconds_per_step"] for k, v in results.items() if v.get("available")}
    return {"results": results, "selected": min(available, key=available.get) if available else None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark hourly equity model devices")
    parser.add_argument("--run", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps", "cuda"])
    args = parser.parse_args(argv)
    print(json.dumps(benchmark(args.run, args.devices), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
