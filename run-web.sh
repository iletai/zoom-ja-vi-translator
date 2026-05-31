#!/usr/bin/env bash
# Launch the live Streamlit dashboard for the Zoom JA->VI translator (macOS / Linux).
#
# The dashboard is decoupled from the audio pipeline: it only *tails* the JSONL
# evidence log the translator writes, so it can never add latency or drop data.
# Start the translator separately with logging, e.g.:
#   ./run.sh --system-audio --log test_audio/evidence/live.jsonl
# then pick that file in the dashboard sidebar.
#
# This script reuses the same arm64-aware venv as run.sh, installs the (tiny)
# web dependency on first run, and opens the dashboard in your browser.
#
# Usage:
#   ./run-web.sh                       # launch dashboard on http://localhost:8501
#   ./run-web.sh --server.port 8600    # extra args pass through to streamlit
set -euo pipefail
cd "$(dirname "$0")"

HOST_ARCH="$(uname -m)"

# Return 0 if the given interpreter runs as native arm64 (not under Rosetta).
is_arm64_python() {
  "$1" -c 'import platform,sys; sys.exit(0 if platform.machine()=="arm64" else 1)' \
    >/dev/null 2>&1
}

PYTHON=""
VENV=".venv"

if [[ "$HOST_ARCH" == "arm64" ]]; then
  for cand in \
    /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
    /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3 \
    /usr/bin/python3; do
    if [[ -x "$cand" ]] && is_arm64_python "$cand"; then
      PYTHON="$cand"
      VENV=".venv-arm64"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  PYTHON="python3"
  for cand in python3.11 python3.12 python3.10 python3.9; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
fi

echo "==> Using $PYTHON ($("$PYTHON" -c 'import platform;print(platform.machine())')) in $VENV"

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating virtual environment ($PYTHON)…"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Install the web dependency once (separate marker from the main ML deps so the
# dashboard works even if the full pipeline deps were never installed).
if [[ ! -f "$VENV/.web_deps_installed" ]]; then
  echo "==> Installing web dependencies (streamlit)…"
  python -m pip install --upgrade pip
  python -m pip install -r requirements-web.txt
  touch "$VENV/.web_deps_installed"
fi

echo "==> Starting Streamlit dashboard → http://localhost:8501"
exec streamlit run webui/streamlit_app.py "$@"
