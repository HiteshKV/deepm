# Top-10 Global Stocks Buy-the-Move Strategy

This package is intentionally separate from DeePM. It is a non-ML, point-in-time
stock backtester for testing a fixed-notional accumulation strategy on the
largest global public common equities by USD market cap.

Run the bundled synthetic sample:

```bash
.venv/bin/python -m stock_top10_strategy validate --config stock_top10_strategy/configs/default.yaml
.venv/bin/python -m stock_top10_strategy backtest --config stock_top10_strategy/configs/default.yaml
```

Build and run the public 2000-current proxy dataset:

```bash
.venv/bin/python -m stock_top10_strategy build-wikipedia-yahoo
.venv/bin/python -m stock_top10_strategy validate --config stock_top10_strategy/configs/wikipedia_yahoo.yaml
.venv/bin/python -m stock_top10_strategy backtest --config stock_top10_strategy/configs/wikipedia_yahoo.yaml
```

The builder writes:

```text
stock_top10_strategy/data/top10_wikipedia_yahoo_2000_current.csv
stock_top10_strategy/data/top10_wikipedia_yahoo_2000_current_audit.json
```

Regenerate a report:

```bash
.venv/bin/python -m stock_top10_strategy report --run stock_top10_strategy/runs/<run_id>
```

If you already have a point-in-time historical membership/market-cap file but
need quick proxy prices for a rough experiment:

```bash
.venv/bin/python -m stock_top10_strategy fill-yahoo-prices \
  --input path/to/point_in_time_membership.csv \
  --output path/to/point_in_time_membership_with_yahoo_prices.csv
```

For production research, replace `data.path` with a point-in-time vendor export
that contains historical market cap, adjusted prices, FX, active/inactive flags,
and stable security IDs. Yahoo-only data is not sufficient for a reliable
historical top-10 universe because it cannot reconstruct membership without
survivorship bias.
