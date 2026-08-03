#!/usr/bin/env bash
set -euo pipefail

LABEL="com.deepm.ibkr-paper"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/${UID}"

launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
if [[ -f "$PLIST_PATH" ]]; then
  mv "$PLIST_PATH" "${PLIST_PATH}.disabled"
fi

echo "Stopped ${LABEL}"
echo "Saved the plist as ${PLIST_PATH}.disabled"
