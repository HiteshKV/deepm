#!/usr/bin/env bash
# Run the current 2026 Yahoo-proxy main DeePM pipeline.
#
# This is intentionally narrower than scripts/reproduce.sh: it trains the
# production DeePM-GAT model, reruns traditional baselines, backtests DeePM-GAT,
# and writes the main comparison CSV for the dated 2026 snapshot.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export WANDB_ENTITY="${WANDB_ENTITY:-hkvelakaturi-e}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    "$PYTHON_BIN" - <<'PY'
import wandb

if not wandb.api.api_key:
    raise SystemExit(
        "WANDB_API_KEY is not set and no existing wandb login was found. "
        "Run `wandb login` or export WANDB_API_KEY before starting."
    )
PY
fi

"$PYTHON_BIN" scripts/validate_data_snapshot.py data/data_20260625.parquet

if [ ! -f data/feats-data_20260625.parquet ]; then
    "$PYTHON_BIN" scripts/prepare_features.py --input data/data_20260625.parquet
fi

"$PYTHON_BIN" -m deepm.training -r deepm-gat -a DeePM
bash scripts/reproduce.sh --baselines
"$PYTHON_BIN" -m deepm.backtest --name bt-deepm-gat --diagnostics

"$PYTHON_BIN" scripts/aggregate_metrics.py \
    --title "Main Results: Yahoo Proxy Data (2010-2026)" \
    --csv backtest_results/current_20260625_main_metrics.csv \
    bt-baseline-longonly \
    bt-baseline-tsmom \
    bt-baseline-tsmom-rm \
    bt-baseline-tsmom-mvo \
    bt-baseline-tsmom-mvo-tp \
    bt-baseline-tsmom-erc \
    bt-baseline-macd \
    bt-baseline-macd-rm \
    bt-baseline-macd-mvo-tp \
    bt-deepm-gat
