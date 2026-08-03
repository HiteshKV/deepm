#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env.live" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.live"
  set +a
fi
