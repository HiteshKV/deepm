# DeRegiME Data

The standalone DeRegiME runner expects processed long-format CSV files under
this directory with columns:

```text
date,cols,data
```

For paper datasets, recreate the processed files from the project root:

```bash
.venv/bin/python scripts/download_tfb_data.py --extract_dir raw_data/tfb
.venv/bin/python scripts/prepare_datasets.py --raw_dir raw_data/tfb --out_dir data
.venv/bin/python scripts/fetch_nasdaq_yfinance.py --out data/NASDAQ_1990_2025.csv
```

For DeePM-aligned rolling DeRegiME experiments, place or generate the DeePM
long-format input at:

```text
data/deregime/deepm_deregime_input_20260625.csv
```
