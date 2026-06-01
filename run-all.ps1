# Run BOTH the translator (with logging) AND the Streamlit web dashboard together.
#
# This script starts the translator capturing system audio in the background,
# then launches the Streamlit dashboard in the foreground. Press Ctrl+C to stop
# both. The dashboard auto-tails the live evidence log.
#
# Usage:
#   ./run-all.ps1                  # translator + web dashboard
#   ./run-all.ps1 -Streaming       # use low-latency streaming ASR
#   ./run-all.ps1 -Llm             # use the local Qwen LLM translation backend
#   ./run-all.ps1 -Port 8600       # custom Streamlit port

param(
    [switch]$Streaming,
    [switch]$Llm,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Enable UTF-8 console output for Japanese/Vietnamese characters.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# Pick a Python 3.9-3.12 interpreter.
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = "py -3.11"
}

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment..."
    & $python -m venv .venv
}

& ".venv\Scripts\Activate.ps1"

# Fix SSL certificate issues on Windows.
$certifi = python -c "import certifi; print(certifi.where())" 2>$null
if ($certifi) {
    $env:SSL_CERT_FILE = $certifi
    $env:REQUESTS_CA_BUNDLE = $certifi
}

# Install all deps (main + web) once.
if (-not (Test-Path ".venv\.deps_installed")) {
    Write-Host "==> Installing dependencies..."
    python -m pip install --upgrade pip
    python -m pip install webrtcvad-wheels
    python -m pip install "ctranslate2==4.5.0" "setuptools<70"
    python -m pip install numpy soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm pip-system-certs certifi
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}
if (-not (Test-Path ".venv\.web_deps_installed")) {
    Write-Host "==> Installing web dependencies (streamlit)..."
    python -m pip install -r requirements-web.txt
    New-Item -ItemType File -Path ".venv\.web_deps_installed" | Out-Null
}

# Download models once.
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading models (first run only, ~2.5 GB)..."
    python scripts\download_models.py
}

if ($Llm) {
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

# HuggingFace offline mode.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Shared log file for this session.
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PSScriptRoot "test_audio\evidence"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir "run_${timestamp}.jsonl"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Zoom JA->VI Translator + Dashboard" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Log:       $logFile"
Write-Host "  Dashboard: http://localhost:$Port"
Write-Host "  Press Ctrl+C to stop both."
Write-Host ""

# Start the translator in a background job.
$translatorArgs = @("main.py", "--system-audio", "--log", $logFile)
if ($Streaming) { $translatorArgs += "--streaming" }

$previousTranslator = $env:ZT_TRANSLATOR
if ($Llm) {
    $env:ZT_TRANSLATOR = "llm"
}

try {
    $translatorJob = Start-Process -FilePath "python" -ArgumentList $translatorArgs `
        -NoNewWindow -PassThru
}
finally {
    if ($Llm) {
        if ($null -eq $previousTranslator) {
            Remove-Item Env:ZT_TRANSLATOR -ErrorAction SilentlyContinue
        }
        else {
            $env:ZT_TRANSLATOR = $previousTranslator
        }
    }
}

Write-Host "[Translator PID $($translatorJob.Id)] Capturing system audio..." -ForegroundColor Green

# Give translator a moment to start writing the log.
Start-Sleep -Seconds 3

# Run Streamlit in the foreground (Ctrl+C stops this).
try {
    streamlit run webui\streamlit_app.py --server.port $Port
} finally {
    # When Streamlit exits (Ctrl+C), also stop the translator.
    if (-not $translatorJob.HasExited) {
        Write-Host "`nStopping translator (PID $($translatorJob.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $translatorJob.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Done. Log saved: $logFile" -ForegroundColor Green
}
