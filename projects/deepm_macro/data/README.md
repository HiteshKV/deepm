# DeePM Macro Data

This directory holds DeePM macro futures/FX research data and feature caches.

Expected current local files:

```text
data_20260625.parquet
data_20260625_sources.csv
data_20260625_sources.json
feats-data_20260625.parquet
```

Regenerate the public Yahoo proxy snapshot and features from this project root:

```bash
.venv/bin/python scripts/build_public_yahoo_data.py \
  --start-date 1990-01-02 \
  --end-date 2026-06-25 \
  --output data/data_20260625.parquet \
  --sources-output data/data_20260625_sources.csv \
  --json-output data/data_20260625_sources.json

.venv/bin/python scripts/prepare_features.py --input data/data_20260625.parquet
```

DeRegiME sidecar feature caches, when used by macro DeePM experiments, live
under `data/deregime/`. The standalone DeRegiME project is in
`../deregime/`.
