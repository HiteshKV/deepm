# Hourly Top-10 Global Equity DeePM Pilot

This standalone project trains DeePM-style hourly equity models on the largest
IBKR-tradable global companies at each point in time. The universe is selected
causally from prior-close USD market cap, then hourly features feed a
multi-input, multi-output model that produces one position per active company.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

## Data

Canonical vendor inputs live under `equity_intraday_deepm/vendor_data/`:

```text
security_master.parquet
market_caps.parquet
hourly_bars.parquet
```

See `VENDOR_DATA.md` for the full schema and recommended institutional data
sources. Free public sources are not sufficient for a correct 2000-current
point-in-time global hourly top-10 dataset.

## Commands

Validate source data:

```bash
.venv/bin/python -m equity_intraday_deepm data validate-source \
  --config equity_intraday_deepm/configs/data_vendor_ibkr_top10.yaml
```

Build universe and features:

```bash
.venv/bin/python -m equity_intraday_deepm data build-universe \
  --config equity_intraday_deepm/configs/data_vendor_ibkr_top10.yaml \
  --start 2000-01-01 \
  --end 2026-07-10 \
  --active-top-n 10

.venv/bin/python -m equity_intraday_deepm data build-features \
  --config equity_intraday_deepm/configs/data_vendor_ibkr_top10.yaml
```

Train, finalize, and backtest a K=10 ensemble:

```bash
WANDB_PROJECT=EQH10_DMN .venv/bin/python -m equity_intraday_deepm.training \
  -r eqh10-xatt -a EQH_DEEPM_XATT --device auto

.venv/bin/python -m equity_intraday_deepm finalize \
  --run eqh10-xatt \
  --top-n 10 \
  --allow-partial

.venv/bin/python -m equity_intraday_deepm.backtest \
  --name bt-eqh10-xatt-k10 \
  --diagnostics
```

Generate the ablation report:

```bash
.venv/bin/python -m equity_intraday_deepm.report \
  --title "Hourly Top-10 Global Equity DeePM Pilot" \
  --output equity_intraday_deepm/reports/hourly_top10_ablation.pdf
```
