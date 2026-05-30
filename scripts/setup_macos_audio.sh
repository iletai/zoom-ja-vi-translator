#!/usr/bin/env bash
# Set up macOS system-audio capture for the Zoom translator.
#
# macOS cannot capture system audio natively, so we install the free BlackHole
# 2ch virtual audio driver and route Zoom's output through it. This script
# installs BlackHole (you will be asked for your admin password) and then prints
# the one-time Audio MIDI Setup steps.
set -euo pipefail

DRIVER="/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"

find_blackhole_pkg() {
  # Prefer the Caskroom copy, fall back to Homebrew's download cache.
  local pkg
  pkg=$(find /usr/local/Caskroom/blackhole-2ch /opt/homebrew/Caskroom/blackhole-2ch \
        -name "*.pkg" 2>/dev/null | head -1)
  if [[ -z "$pkg" ]]; then
    pkg=$(find "$HOME/Library/Caches/Homebrew/downloads" -iname "*BlackHole2ch*.pkg" 2>/dev/null | head -1)
  fi
  printf '%s' "$pkg"
}

echo "==> Installing BlackHole 2ch (virtual audio cable)…"
if [[ -d "$DRIVER" ]]; then
  echo "    BlackHole driver already present at $DRIVER. Skipping."
else
  # Make sure the .pkg is downloaded (brew may already record it as installed
  # even when the driver was never actually placed — so we don't rely on that).
  if command -v brew >/dev/null 2>&1; then
    brew install blackhole-2ch >/dev/null 2>&1 || true
  fi

  PKG="$(find_blackhole_pkg)"
  if [[ -z "$PKG" || ! -f "$PKG" ]]; then
    echo "ERROR: Could not find the BlackHole .pkg installer." >&2
    echo "       Try: brew reinstall --cask blackhole-2ch" >&2
    exit 1
  fi

  echo "    Running the system installer (you will be asked for your admin password)…"
  echo "    pkg: $PKG"
  sudo installer -pkg "$PKG" -target /

  # Reload Core Audio so the new device shows up without a reboot.
  echo "    Restarting Core Audio…"
  sudo pkill -9 coreaudiod 2>/dev/null || true
  sleep 2

  if [[ -d "$DRIVER" ]]; then
    echo "    ✅ BlackHole installed successfully."
  else
    echo "    ⚠️  Driver still not detected. A logout/login or reboot may be needed." >&2
  fi
fi

cat <<'STEPS'

==> One-time routing setup (so you hear audio AND the app can capture it)

1. Open "Audio MIDI Setup" (Applications ▸ Utilities).
2. Click "+" (bottom-left) ▸ "Create Multi-Output Device".
3. In the Multi-Output Device, tick BOTH:
       [x] Your speakers / headphones   (so you still hear the meeting)
       [x] BlackHole 2ch                (so the app can capture it)
   Tip: set your real output as the "Primary/Master" device.
4. System Settings ▸ Sound ▸ Output ▸ select the "Multi-Output Device".
5. In Zoom: Settings ▸ Audio ▸ Speaker ▸ leave as System Default
   (it follows the Multi-Output Device).

==> Then run the translator capturing system audio:

       source .venv/bin/activate
       python3 main.py --system-audio

   Verify the captured device with:

       python3 main.py --list-devices      # you should see "BlackHole 2ch"

Grant your terminal Microphone permission if prompted
(System Settings ▸ Privacy & Security ▸ Microphone).
STEPS
