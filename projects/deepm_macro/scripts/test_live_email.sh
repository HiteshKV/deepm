#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ibkr-paper}"
DATE_ARG="${2:-2026-06-25}"
CONFIG="${DEEPM_LIVE_CONFIG:-configs/live/deepm_gat_ibkr.yaml}"

cd "$(dirname "$0")/.."

source scripts/live_env.sh
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif [[ -x "../../.venv/bin/python" ]]; then
    PYTHON_BIN="../../.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

exec "$PYTHON_BIN" -m deepm.live.email_test \
  --config "$CONFIG" \
  --mode "$MODE" \
  --date "$DATE_ARG"
