# Run the Zoom JA->VI translator on Windows (PowerShell).
#
# Windows captures system audio natively via WASAPI loopback - no virtual cable
# needed. This script creates/activates the venv, installs dependencies and
# downloads models on first run, then starts the translator on system audio.
#
# Usage:
#   ./run.ps1                 # capture Zoom/system audio (recommended)
#   ./run.ps1 -ListDevices    # list audio devices and exit
#   ./run.ps1 -Mic            # capture the default microphone instead

param(
    [switch]$ListDevices,
    [switch]$Mic
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Pick a Python 3.9-3.12 interpreter (ML wheels are not built for 3.13+ yet).
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = "py -3.11"
}

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment..."
    & $python -m venv .venv
}

& ".venv\Scripts\Activate.ps1"

# Install deps once (marker file avoids reinstalling every run).
if (-not (Test-Path ".venv\.deps_installed")) {
    Write-Host "==> Installing dependencies..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}

# Download models once.
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading models (first run only)..."
    python scripts\download_models.py
}

if ($ListDevices) {
    python main.py --list-devices
}
elseif ($Mic) {
    python main.py
}
else {
    # Default speaker loopback is selected automatically on Windows.
    python main.py --system-audio
}
