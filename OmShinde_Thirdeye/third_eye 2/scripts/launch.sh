#!/usr/bin/env bash
set -euo pipefail

# Launch the Third Eye runtime in headless mode with audio announcements disabled
# This is container-friendly and avoids TTS dependencies in minimal images

cd "$(dirname "$0")/.."

# Optional profile selection (low-power, balanced, high-performance)
export THIRD_EYE_PROFILE="${THIRD_EYE_PROFILE:-balanced}"

# Create runtime directories
mkdir -p logs recordings

# Run the main runtime in headless mode. TTS is enabled by default unless overridden.
# Prefer project venv Python if available; fallback to system python3
VENV_PY="$(pwd)/.venv/bin/python"
if [ -x "$VENV_PY" ]; then
  exec "$VENV_PY" -u main.py --headless "$@"
else
  exec python3 -u main.py --headless "$@"
fi