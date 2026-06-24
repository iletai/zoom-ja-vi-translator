#!/usr/bin/env bash
# Launch the Speaksy host bridge: serves the React UI (webui/rd_ui_v1.1.html)
# and speaks its WebSocket protocol so the same UI runs in any browser.
#
# The host is zero-dependency (Python stdlib only), so this needs nothing
# installed. If a project venv exists (.venv / .venv-arm64), it is used so the
# host can probe *real* audio devices via soundcard; otherwise it falls back to
# a demo device list — the UI works either way.
#
# Usage:
#   ./run-host.sh                 # serve on http://127.0.0.1:8770
#   ./run-host.sh --port 8888     # extra args pass through to host_server.py
#   ./run-host.sh --host 0.0.0.0  # expose on LAN (use with care)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="python3"
for VENV in .venv-arm64 .venv; do
  if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
    PYTHON="python"
    echo "==> Using venv: $VENV (real device probing enabled)"
    break
  fi
done

echo "==> Starting Speaksy host → http://127.0.0.1:8770 (Ctrl+C to stop)"
exec "$PYTHON" webui/host_server.py "$@"
