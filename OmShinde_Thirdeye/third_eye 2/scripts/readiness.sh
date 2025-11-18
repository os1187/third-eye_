#!/usr/bin/env bash
set -euo pipefail
# Readiness probe for Third Eye shortcuts server
# Usage:
#   SHORTCUTS_URL=http://127.0.0.1:8765/ready ./scripts/readiness.sh
# Optional:
#   SHORTCUTS_AUTH=user:pass   # Basic auth credential if required
#   CURL_FLAGS="-k"            # Extra curl flags

URL="${SHORTCUTS_URL:-http://127.0.0.1:8765/ready}"
AUTH="${SHORTCUTS_AUTH:-}"
EXTRA_CURL_FLAGS=${CURL_FLAGS:-}

if [[ -n "${AUTH}" ]]; then
  AUTH_FLAG=( -u "${AUTH}" )
else
  AUTH_FLAG=()
fi

# Get both HTTP code and body (for optional JSON check)
RESP=$(mktemp)
trap 'rm -f "$RESP"' EXIT
HTTP_CODE=$(curl -s -o "$RESP" -w "%{http_code}" "${AUTH_FLAG[@]}" ${EXTRA_CURL_FLAGS} "${URL}")

if [[ "${HTTP_CODE}" -eq 200 ]]; then
  # If body contains "ready": true consider ready; otherwise still OK since 200 indicates readiness
  if grep -q '"ready"\s*:\s*true' "$RESP" 2>/dev/null; then
    echo "READY: endpoint returned 200 and ready=true"
  else
    echo "READY: endpoint returned 200"
  fi
  exit 0
else
  echo "NOT READY: endpoint HTTP ${HTTP_CODE}"
  cat "$RESP" || true
  exit 1
fi