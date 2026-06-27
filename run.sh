#!/usr/bin/env bash
# Run the Zoom JA->VI translator on macOS / Linux.
#
# Creates/activates the venv, installs dependencies and downloads models on the
# first run, then starts the translator capturing system audio.
#
# On Apple Silicon (M1/M2/M3) this AUTOMATICALLY uses a native arm64 Python and a
# separate .venv-arm64 environment. Running natively instead of under Rosetta 2
# makes translation ~3x faster (NLLB MT ~1s vs ~3s per sentence) for free.
#
# Usage:
#   ./run.sh                 # capture system audio (needs BlackHole on macOS)
#   ./run.sh --list-devices  # list audio devices and exit
#   ./run.sh --mic           # capture the default microphone instead
#   ./run.sh --streaming     # low-latency live captions (extra args pass through)
#   ./run.sh --router        # translate via the 9router gateway (bootstraps .env)
set -euo pipefail
cd "$(dirname "$0")"

# --router: offload translation to the local 9router gateway. main.py doesn't
# know this flag (strict argparse), so consume it here, set the backend env, and
# bootstrap .env from the template on first use so ZT_ROUTER_KEY/MODEL are set.
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--router" ]]; then
    export ZT_TRANSLATOR=router
    if [[ ! -f .env && -f .env.example ]]; then
      cp .env.example .env
      echo "==> Created .env from .env.example — edit ZT_ROUTER_KEY if your gateway needs it."
    fi
  else
    ARGS+=("$arg")
  fi
done
# Re-set positional params to the filtered list (empty-array-safe on bash 3.2).
set -- ${ARGS[@]+"${ARGS[@]}"}

HOST_ARCH="$(uname -m)"

# Return 0 if the given interpreter runs as native arm64 (not under Rosetta).
is_arm64_python() {
  "$1" -c 'import platform,sys; sys.exit(0 if platform.machine()=="arm64" else 1)' \
    >/dev/null 2>&1
}

PYTHON=""
VENV=".venv"

if [[ "$HOST_ARCH" == "arm64" ]]; then
  # Prefer a native arm64 interpreter so the ML stack doesn't run under Rosetta.
  # Homebrew (arm64) lives in /opt/homebrew; the macOS system python at
  # /usr/bin/python3 is a universal binary that runs arm64 natively here.
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
  if [[ -z "$PYTHON" ]]; then
    echo "==> No native arm64 Python found; falling back to Rosetta (slower)." >&2
  fi
fi

if [[ -z "$PYTHON" ]]; then
  # Non-Apple-Silicon hosts (Intel mac / Linux): pick a 3.9-3.12 interpreter
  # (ML wheels are not built for 3.13+ yet).
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

if [[ ! -f "$VENV/.deps_installed" ]]; then
  echo "==> Installing dependencies…"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  touch "$VENV/.deps_installed"
fi

if [[ ! -d "models/reazonspeech-k2-v2" ]]; then
  echo "==> Downloading models (first run only)…"
  python scripts/download_models.py
fi

case "${1:-}" in
  --list-devices) python main.py --list-devices ;;
  --mic)          python main.py ;;
  *)              python main.py --system-audio "$@" ;;
esac
