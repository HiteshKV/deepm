#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ibkr-paper}"
DATE_ARG="${2:-2026-06-25}"
CONFIG="${DEEPM_LIVE_CONFIG:-configs/live/deepm_gat_ibkr.yaml}"

cd "$(dirname "$0")/.."

source scripts/live_env.sh

exec .venv/bin/python -m deepm.live.email_test \
  --config "$CONFIG" \
  --mode "$MODE" \
  --date "$DATE_ARG"
