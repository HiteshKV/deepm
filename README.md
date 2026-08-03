# Research Projects Workspace

This repository is organized as four standalone Python projects. Each project
has its own package, configs, scripts, tests, README, and dependency files.

## Projects

- `projects/deepm_macro/` - original DeePM macro portfolio model, macro futures
  data pipeline, seed ensembles, backtests, reports, and IBKR paper/live daily
  management.
- `projects/deregime/` - standalone DeRegiME probabilistic regime forecasting
  model and DeePM-aligned rolling experiment helpers.
- `projects/stock_top10_strategy/` - non-ML top-10 global equity buy-the-move
  backtester for configurable +/- move accumulation strategies.
- `projects/equity_intraday_deepm/` - hourly top-10 global equity DeePM-style
  pilot with point-in-time universe construction, hourly features, training,
  ensemble backtesting, and reporting.

## Setup Pattern

Each project is designed to work after being copied out of this repository.

```bash
cd projects/<project>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m unittest discover -s tests
```

Generated data, model checkpoints, logs, and run outputs are intentionally
project-local and git-ignored. See each project README and `data/README.md` for
the expected data sources and regeneration commands.
