#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ibkr-paper}"
CONFIG="${DEEPM_LIVE_CONFIG:-configs/live/deepm_gat_ibkr.yaml}"
SEND_EMAIL="${DEEPM_LIVE_SEND_EMAIL:-1}"

cd "$(dirname "$0")/.."

source scripts/live_env.sh
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
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

mkdir -p live_state/logs

ARGS=(--mode "$MODE" --config "$CONFIG")
if [[ "$SEND_EMAIL" == "1" ]]; then
  ARGS+=(--send-email)
fi

PID_FILE="live_state/${MODE}-daemon.pid"
"$PYTHON_BIN" -m deepm.live.daemon "${ARGS[@]}" &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"

cleanup() {
  if kill -0 "$DAEMON_PID" 2>/dev/null; then
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  if [[ -f "$PID_FILE" ]] && [[ "$(tr -d '[:space:]' < "$PID_FILE")" == "$DAEMON_PID" ]]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT HUP INT TERM

wait "$DAEMON_PID"
