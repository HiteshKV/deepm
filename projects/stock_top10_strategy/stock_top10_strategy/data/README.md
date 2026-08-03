# Stock Top-10 Strategy Data

The strategy expects point-in-time daily equity rows with stable security IDs,
market cap, adjusted prices, FX, and active/delist flags.

The bundled public proxy builder writes:

```text
top10_wikipedia_yahoo_2000_current.csv
top10_wikipedia_yahoo_2000_current_audit.json
```

Run from the project root:

```bash
.venv/bin/python -m stock_top10_strategy build-wikipedia-yahoo
```

Yahoo/Wikipedia data is a proxy only. For serious research, replace the config
`data.path` with a vendor point-in-time export.
