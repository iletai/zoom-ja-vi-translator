#!/usr/bin/env bash
# Launch the Speaksy host with overlay subtitle window + auto-start + history.
#
# Equivalent to: ./run-host.sh --overlay --auto-start --history
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
for VENV in .venv-arm64 .venv; do
  if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
    PYTHON="python"
    echo "==> Using venv: $VENV (real device probing enabled)"
    break
  fi
done

echo "==> Starting Speaksy host with overlay + history"
echo "    UI  → http://127.0.0.1:8770"
echo "    Log → ~/.speaksy/history/ (set ZT_HISTORY_DIR to customise)"
exec "$PYTHON" webui/host_server.py --overlay --auto-start --history "$@"
