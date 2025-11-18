#!/usr/bin/env bash
set -euo pipefail
# Healthcheck script for Third Eye shortcuts server
# Usage:
#   SHORTCUTS_URL=http://127.0.0.1:8765/health ./scripts/healthcheck.sh
# Optional:
#   SHORTCUTS_AUTH=user:pass   # Basic auth credential if required
#   CURL_FLAGS="-k"            # Extra curl flags (e.g., allow self-signed)

URL="${SHORTCUTS_URL:-http://127.0.0.1:8765/health}"
AUTH="${SHORTCUTS_AUTH:-}"
EXTRA_CURL_FLAGS=${CURL_FLAGS:-}

if [[ -n "${AUTH}" ]]; then
  AUTH_FLAG=( -u "${AUTH}" )
else
  AUTH_FLAG=()
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${AUTH_FLAG[@]}" ${EXTRA_CURL_FLAGS} "${URL}")
if [[ "${HTTP_CODE}" -eq 200 ]]; then
  echo "OK: health endpoint returned 200"
  exit 0
else
  echo "ERROR: health endpoint HTTP ${HTTP_CODE}"
  exit 1
fi