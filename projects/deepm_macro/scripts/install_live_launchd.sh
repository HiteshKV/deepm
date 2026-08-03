#!/usr/bin/env bash
set -euo pipefail

LABEL="com.deepm.ibkr-paper"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${LAUNCH_AGENTS_DIR}/${LABEL}.plist"
LOG_DIR="${PROJECT_DIR}/live_state/logs"
DOMAIN="gui/${UID}"

mkdir -p "$LAUNCH_AGENTS_DIR" "$LOG_DIR"

DEEPM_PROJECT_DIR="$PROJECT_DIR" DEEPM_PLIST_PATH="$PLIST_PATH" /usr/bin/python3 - <<'PY'
import os
import plistlib
from pathlib import Path

project = Path(os.environ["DEEPM_PROJECT_DIR"])
plist_path = Path(os.environ["DEEPM_PLIST_PATH"])
log_dir = project / "live_state" / "logs"
payload = {
    "Label": "com.deepm.ibkr-paper",
    "ProgramArguments": [
        "/bin/bash",
        str(project / "scripts" / "run_live_daemon.sh"),
        "ibkr-paper",
    ],
    "WorkingDirectory": str(project),
    "RunAtLoad": True,
    "KeepAlive": True,
    "ProcessType": "Background",
    "ThrottleInterval": 30,
    "StandardOutPath": str(log_dir / "ibkr-paper-launchd.log"),
    "StandardErrorPath": str(log_dir / "ibkr-paper-launchd.err.log"),
    "EnvironmentVariables": {
        "DEEPM_LIVE_SEND_EMAIL": "1",
        "PYTHONUNBUFFERED": "1",
    },
}
with plist_path.open("wb") as handle:
    plistlib.dump(payload, handle, sort_keys=False)
PY

/usr/bin/plutil -lint "$PLIST_PATH"
launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST_PATH"
launchctl enable "${DOMAIN}/${LABEL}"
launchctl kickstart -k "${DOMAIN}/${LABEL}"

SERVICE_INFO="$(launchctl print "${DOMAIN}/${LABEL}")"
if ! grep -q "state = running" <<<"$SERVICE_INFO"; then
  launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
  echo "LaunchAgent did not remain running." >&2
  echo "If this project is under Documents, macOS privacy may deny background access." >&2
  echo "Move the checkout outside Documents or grant the required shell/Python binaries" >&2
  echo "Full Disk Access, then rerun this installer." >&2
  echo "Details: ${LOG_DIR}/ibkr-paper-launchd.err.log" >&2
  exit 1
fi

echo "Installed and started ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Status: launchctl print ${DOMAIN}/${LABEL}"
echo "Log: ${LOG_DIR}/ibkr-paper-launchd.log"
