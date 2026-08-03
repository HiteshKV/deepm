# Hourly Top-10 Equity Data Requirements

This project cannot produce a fully correct 2000-current global hourly top-10
equity dataset from public free sources. The canonical files require
institutional point-in-time data:

- `security_master.parquet`: permanent security ID, ticker history, company
  name, country, sector, currency, exchange, IBKR `conId`, listing dates,
  security type, and tradability.
- `market_caps.parquet`: daily point-in-time USD market cap for all globally
  eligible common equities, or the raw split-adjusted price, shares outstanding,
  and FX needed to compute it.
- `hourly_bars.parquet`: hourly OHLCV, adjusted close, and FX-to-USD for every
  security that can enter the top 10, including after it leaves so exits can be
  priced.

Recommended sources:

- LSEG Tick History for global intraday trades/quotes/bars back to 1996.
- FactSet Tick History for global exchange-normalized intraday data.
- FactSet/LSEG/Bloomberg point-in-time security master, corporate actions,
  shares outstanding, FX, and identifier history.
- WRDS/Compustat Global can help build daily market caps, but it does not
  replace global hourly bars.

Not sufficient for canonical results:

- Yahoo/yfinance: hourly history is short and lacks point-in-time global
  security master.
- Wikipedia rankings: sparse ranking snapshots, not daily market cap history.
- IBKR API alone: good for execution and recent/current checks, not bulk
  2000-current global research backfill.

Import vendor exports:

```bash
.venv/bin/python -m equity_intraday_deepm.vendor_import sources

.venv/bin/python -m equity_intraday_deepm.vendor_import build-canonical \
  --config equity_intraday_deepm/configs/vendor_build_template.yaml

.venv/bin/python -m equity_intraday_deepm.vendor_import import-vendor \
  --config equity_intraday_deepm/configs/vendor_import_template.yaml
```

The importer writes the three canonical parquet files under
`equity_intraday_deepm/vendor_data/`, builds the causal top-10 universe, and
fails if strict top-10 coverage is incomplete.

`build-canonical` is the preferred route. It computes daily USD market caps from
daily prices, point-in-time shares outstanding, and FX, consolidates share
classes to company-level market cap, selects the top 10 companies using prior
close data, and then maps each company to its primary IBKR-tradable security.

`import-vendor` is for vendors that already export clean daily
`market_cap_usd`.
