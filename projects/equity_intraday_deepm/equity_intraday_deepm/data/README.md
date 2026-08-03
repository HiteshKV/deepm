# Hourly Equity DeePM Data

Generated universe, feature, and audit files for the hourly top-10 equity pilot
live here.

Canonical source files are expected under `equity_intraday_deepm/vendor_data/`:

```text
security_master.parquet
market_caps.parquet
hourly_bars.parquet
```

Build the derived data from the project root:

```bash
.venv/bin/python -m equity_intraday_deepm data build-universe \
  --config equity_intraday_deepm/configs/data_vendor_ibkr_top10.yaml \
  --start 2000-01-01 \
  --end 2026-07-10 \
  --active-top-n 10

.venv/bin/python -m equity_intraday_deepm data build-features \
  --config equity_intraday_deepm/configs/data_vendor_ibkr_top10.yaml
```
