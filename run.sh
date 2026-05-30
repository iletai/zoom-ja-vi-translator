#!/usr/bin/env bash
# Run the Zoom JA->VI translator on macOS / Linux.
#
# Creates/activates the venv, installs dependencies and downloads models on the
# first run, then starts the translator capturing system audio.
#
# Usage:
#   ./run.sh                 # capture system audio (needs BlackHole on macOS)
#   ./run.sh --list-devices  # list audio devices and exit
#   ./run.sh --mic           # capture the default microphone instead
set -euo pipefail
cd "$(dirname "$0")"

# Prefer a Python 3.9-3.12 interpreter (ML wheels are not built for 3.13+ yet).
PYTHON="python3"
for cand in python3.11 python3.12 python3.10 python3.9; do
  if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
done

if [[ ! -d ".venv" ]]; then
  echo "==> Creating virtual environment ($PYTHON)…"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -f ".venv/.deps_installed" ]]; then
  echo "==> Installing dependencies…"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch .venv/.deps_installed
fi

if [[ ! -d "models/reazonspeech-k2-v2" ]]; then
  echo "==> Downloading models (first run only)…"
  python scripts/download_models.py
fi

case "${1:-}" in
  --list-devices) python main.py --list-devices ;;
  --mic)          python main.py ;;
  *)              python main.py --system-audio ;;
esac
