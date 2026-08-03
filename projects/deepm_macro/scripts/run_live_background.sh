#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
SESSION_NAME="deepm_paper"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${PROJECT_DIR}/live_state/logs/ibkr-paper-screen.log"
PID_FILE="${PROJECT_DIR}/live_state/ibkr-paper-daemon.pid"

session_exists() {
  local sessions
  sessions="$(screen -ls 2>&1 || true)"
  grep -Eq "[0-9]+\\.${SESSION_NAME}[[:space:]]" <<<"$sessions"
}

daemon_pid() {
  if [[ -f "$PID_FILE" ]]; then
    tr -d '[:space:]' < "$PID_FILE"
  fi
}

daemon_exists() {
  local pid
  pid="$(daemon_pid)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

case "$ACTION" in
  start)
    if session_exists || daemon_exists; then
      echo "already_running"
      exit 0
    fi
    mkdir -p "${PROJECT_DIR}/live_state/logs"
    cd "$PROJECT_DIR"
    screen -dmS "$SESSION_NAME" /bin/bash scripts/run_live_daemon.sh ibkr-paper
    screen -S "$SESSION_NAME" -X logfile "$LOG_FILE"
    screen -S "$SESSION_NAME" -X log on
    echo "started"
    echo "Status: bash scripts/run_live_background.sh status"
    echo "Log: ${LOG_FILE}"
    ;;
  status)
    if session_exists; then
      echo "screen=running"
    else
      echo "screen=stopped"
    fi
    if [[ -f "$PID_FILE" ]]; then
      DAEMON_PID="$(tr -d '[:space:]' < "$PID_FILE")"
      if kill -0 "$DAEMON_PID" 2>/dev/null; then
        echo "daemon_pid=${DAEMON_PID}"
      else
        echo "daemon_pid=${DAEMON_PID} (stale)"
      fi
    fi
    ;;
  stop)
    if daemon_exists; then
      DAEMON_PID="$(daemon_pid)"
      kill "$DAEMON_PID"
    fi
    if session_exists; then
      screen -S "$SESSION_NAME" -X quit
    fi
    echo "stopped"
    ;;
  *)
    echo "Usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
