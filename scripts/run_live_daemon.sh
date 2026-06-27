#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ibkr-paper}"
CONFIG="${DEEPM_LIVE_CONFIG:-configs/live/deepm_gat_ibkr.yaml}"
SEND_EMAIL="${DEEPM_LIVE_SEND_EMAIL:-1}"

cd "$(dirname "$0")/.."

source scripts/live_env.sh

ARGS=(--mode "$MODE" --config "$CONFIG")
if [[ "$SEND_EMAIL" == "1" ]]; then
  ARGS+=(--send-email)
fi

exec .venv/bin/python -m deepm.live.daemon "${ARGS[@]}"
