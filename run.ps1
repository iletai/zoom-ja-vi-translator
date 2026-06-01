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
#   ./run.ps1 -Log            # capture with evidence logging enabled
#   ./run.ps1 -Llm            # use the local Qwen LLM translation backend

param(
    [switch]$ListDevices,
    [switch]$Mic,
    [switch]$Log,
    [switch]$Llm
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

# Fix SSL certificate issues on Windows (corporate proxies, missing CA certs).
$certifi = python -c "import certifi; print(certifi.where())" 2>$null
if ($certifi) {
    $env:SSL_CERT_FILE = $certifi
    $env:REQUESTS_CA_BUNDLE = $certifi
}

# Install deps once (marker file avoids reinstalling every run).
if (-not (Test-Path ".venv\.deps_installed")) {
    Write-Host "==> Installing dependencies..."
    python -m pip install --upgrade pip
    # webrtcvad requires C++ Build Tools; use pre-built wheels instead.
    python -m pip install webrtcvad-wheels
    # Pin ctranslate2 to 4.5.0 (4.7.x crashes with pre-converted models).
    python -m pip install "ctranslate2==4.5.0" "setuptools<70"
    # Install remaining deps.
    python -m pip install numpy soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm pip-system-certs certifi
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}

# Download models once.
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading models (first run only, ~2.5 GB)..."
    python scripts\download_models.py
}

if ($Llm) {
    $env:ZT_TRANSLATOR = "llm"
    # Prefer 1.5B model (lighter, better for 16GB RAM systems); fall back to 3B
    $model1p5b = Join-Path $PSScriptRoot "models\qwen2.5-1.5b-instruct\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    $model3b = Join-Path $PSScriptRoot "models\qwen2.5-3b-instruct\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    if (Test-Path $model1p5b) {
        Write-Host "LLM translation mode active (Qwen2.5-1.5B)" -ForegroundColor Cyan
    } elseif (Test-Path $model3b) {
        Write-Host "LLM translation mode active (Qwen2.5-3B)" -ForegroundColor Cyan
    } else {
        Write-Host "LLM model not found. Downloading Qwen2.5-1.5B (~0.9GB)..." -ForegroundColor Yellow
        python scripts\download_qwen_model.py --size 1.5b
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to download LLM model." -ForegroundColor Red
            exit 1
        }
        Write-Host "LLM translation mode active (Qwen2.5-1.5B)" -ForegroundColor Cyan
    }
}

# HuggingFace offline mode (no network needed after models are downloaded).
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Enable UTF-8 console output for Japanese/Vietnamese characters.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

if ($ListDevices) {
    python main.py --list-devices
}
elseif ($Mic) {
    if ($Log) { python main.py --log } else { python main.py }
}
else {
    # Default speaker loopback is selected automatically on Windows.
    if ($Log) {
        python main.py --system-audio --log
    } else {
        python main.py --system-audio
    }
}
