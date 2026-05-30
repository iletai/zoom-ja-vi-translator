#!/usr/bin/env bash
# Set up macOS system-audio capture for the Zoom translator.
#
# macOS cannot capture system audio natively, so we install the free BlackHole
# 2ch virtual audio driver and route Zoom's output through it. This script
# installs BlackHole (you will be asked for your admin password) and then prints
# the one-time Audio MIDI Setup steps.
set -euo pipefail

echo "==> Installing BlackHole 2ch (virtual audio cable)…"
if [[ -d "/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver" ]]; then
  echo "    BlackHole already installed. Skipping."
else
  if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew not found. Install it from https://brew.sh first." >&2
    exit 1
  fi
  # Reuses the cached .pkg if already downloaded. Prompts for your sudo password.
  brew install blackhole-2ch
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
