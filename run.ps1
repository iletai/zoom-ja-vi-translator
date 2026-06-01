# Run the Zoom JA->VI translator on Windows (PowerShell).
#
# Uses WASAPI loopback to capture system audio natively — no virtual cable needed.
# First run: creates venv, installs deps, downloads models automatically.
#
# Usage:
#   ./run.ps1                 # LLM mode (default, best quality for IT meetings)
#   ./run.ps1 -Nllb           # use NLLB-600M (faster, less context-aware)
#   ./run.ps1 -ListDevices    # list audio devices and exit
#   ./run.ps1 -Mic            # capture microphone instead of system audio
#   ./run.ps1 -Log            # enable evidence logging to JSONL
#   ./run.ps1 -Streaming      # use streaming ASR (lower latency)

param(
    [switch]$ListDevices,
    [switch]$Mic,
    [switch]$Log,
    [switch]$Nllb,
    [switch]$Streaming
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ─── Python interpreter ───────────────────────────────────────────────────
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = "py -3.11"
}

# ─── Virtual environment ──────────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment..." -ForegroundColor Yellow
    & $python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"

# ─── SSL fix (corporate proxies / missing CA certs) ───────────────────────
$certifi = python -c "import certifi; print(certifi.where())" 2>$null
if ($certifi) {
    $env:SSL_CERT_FILE = $certifi
    $env:REQUESTS_CA_BUNDLE = $certifi
}

# ─── Install dependencies (first run only) ────────────────────────────────
if (-not (Test-Path ".venv\.deps_installed")) {
    Write-Host "==> Installing dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip -q
    python -m pip install webrtcvad-wheels -q
    python -m pip install "ctranslate2==4.5.0" "setuptools<70" -q
    python -m pip install "llama-cpp-python>=0.2.90,<0.3.0" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -q
    python -m pip install numpy soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm pip-system-certs certifi -q
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}

# ─── Download ASR models (first run only) ─────────────────────────────────
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading ASR models (~2.5 GB, first run only)..." -ForegroundColor Yellow
    python scripts\download_models.py
}

# ─── Translation backend selection ────────────────────────────────────────
if ($Nllb) {
    $env:ZT_TRANSLATOR = "nllb"
    $backendName = "NLLB-600M (fast, no context)"
} else {
    # Default: LLM mode (better quality for IT meetings)
    $env:ZT_TRANSLATOR = "llm"

    $model1p5b = Join-Path $PSScriptRoot "models\qwen2.5-1.5b-instruct\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    $model3b = Join-Path $PSScriptRoot "models\qwen2.5-3b-instruct\Qwen2.5-3B-Instruct-Q4_K_M.gguf"

    if (Test-Path $model1p5b) {
        $backendName = "Qwen2.5-1.5B (context-aware, IT-optimized)"
    } elseif (Test-Path $model3b) {
        $backendName = "Qwen2.5-3B (context-aware, IT-optimized)"
    } else {
        Write-Host "==> Downloading LLM model (~0.9 GB, first run only)..." -ForegroundColor Yellow
        python scripts\download_qwen_model.py --size 1.5b
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Download failed. Falling back to NLLB." -ForegroundColor Red
            $env:ZT_TRANSLATOR = "nllb"
            $backendName = "NLLB-600M (fallback)"
        } else {
            $backendName = "Qwen2.5-1.5B (context-aware, IT-optimized)"
        }
    }
}

# ─── Environment tuning ──────────────────────────────────────────────────
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# ─── Show config ─────────────────────────────────────────────────────────
if (-not $ListDevices) {
    Write-Host ""
    Write-Host "  Zoom JA->VI Translator" -ForegroundColor Cyan
    Write-Host "  ──────────────────────" -ForegroundColor DarkGray
    Write-Host "  Translator : $backendName" -ForegroundColor White
    Write-Host "  ASR        : ReazonSpeech k2-v2 + hotwords" -ForegroundColor White
    Write-Host "  Audio      : $(if ($Mic) {'Microphone'} else {'System audio (WASAPI loopback)'})" -ForegroundColor White
    if ($Streaming) {
        Write-Host "  Mode       : Streaming (low-latency)" -ForegroundColor White
    }
    Write-Host ""
}

# ─── Run ──────────────────────────────────────────────────────────────────
$mainArgs = @()
if ($Streaming) { $mainArgs += "--streaming" }

if ($ListDevices) {
    python main.py --list-devices
} elseif ($Mic) {
    if ($Log) { $mainArgs += "--log" }
    python main.py @mainArgs
} else {
    $mainArgs += "--system-audio"
    if ($Log) { $mainArgs += "--log" }
    python main.py @mainArgs
}
