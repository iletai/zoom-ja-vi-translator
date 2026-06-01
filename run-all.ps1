# Run BOTH the translator (with logging) AND the Streamlit web dashboard together.
#
# Starts the translator capturing system audio in the background, then launches
# the Streamlit dashboard in the foreground. Press Ctrl+C to stop both.
#
# Usage:
#   ./run-all.ps1                  # LLM translator + web dashboard (default)
#   ./run-all.ps1 -Nllb            # use NLLB-600M instead of LLM
#   ./run-all.ps1 -Streaming       # use low-latency streaming ASR
#   ./run-all.ps1 -Port 8600       # custom Streamlit port

param(
    [switch]$Streaming,
    [switch]$Nllb,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ─── UTF-8 console ────────────────────────────────────────────────────────
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

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

# ─── SSL fix ──────────────────────────────────────────────────────────────
$certifi = python -c "import certifi; print(certifi.where())" 2>$null
if ($certifi) {
    $env:SSL_CERT_FILE = $certifi
    $env:REQUESTS_CA_BUNDLE = $certifi
}

# ─── Install dependencies ─────────────────────────────────────────────────
if (-not (Test-Path ".venv\.deps_installed")) {
    Write-Host "==> Installing dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip -q
    python -m pip install webrtcvad-wheels -q
    python -m pip install "ctranslate2==4.5.0" "setuptools<70" -q
    python -m pip install "llama-cpp-python>=0.2.90,<0.3.0" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -q
    python -m pip install numpy soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm pip-system-certs certifi -q
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}
if (-not (Test-Path ".venv\.web_deps_installed")) {
    Write-Host "==> Installing web dependencies (streamlit)..." -ForegroundColor Yellow
    python -m pip install -r requirements-web.txt -q
    New-Item -ItemType File -Path ".venv\.web_deps_installed" | Out-Null
}

# ─── Download ASR models ──────────────────────────────────────────────────
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading ASR models (~2.5 GB)..." -ForegroundColor Yellow
    python scripts\download_models.py
}

# ─── Translation backend ──────────────────────────────────────────────────
if ($Nllb) {
    $env:ZT_TRANSLATOR = "nllb"
    $backendName = "NLLB-600M"
} else {
    $env:ZT_TRANSLATOR = "llm"
    $model1p5b = Join-Path $PSScriptRoot "models\qwen2.5-1.5b-instruct\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    $model3b = Join-Path $PSScriptRoot "models\qwen2.5-3b-instruct\Qwen2.5-3B-Instruct-Q4_K_M.gguf"

    if (Test-Path $model1p5b) {
        $backendName = "Qwen2.5-1.5B"
    } elseif (Test-Path $model3b) {
        $backendName = "Qwen2.5-3B"
    } else {
        Write-Host "==> Downloading LLM model (~0.9 GB)..." -ForegroundColor Yellow
        python scripts\download_qwen_model.py --size 1.5b
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Download failed. Falling back to NLLB." -ForegroundColor Red
            $env:ZT_TRANSLATOR = "nllb"
            $backendName = "NLLB-600M (fallback)"
        } else {
            $backendName = "Qwen2.5-1.5B"
        }
    }
}

# ─── HuggingFace offline ──────────────────────────────────────────────────
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# ─── Session setup ────────────────────────────────────────────────────────
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PSScriptRoot "test_audio\evidence"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir "run_${timestamp}.jsonl"

Write-Host ""
Write-Host "  ╔══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║  Zoom JA→VI Translator + Dashboard  ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Translator : $backendName" -ForegroundColor White
Write-Host "  ASR        : ReazonSpeech k2-v2 + hotwords" -ForegroundColor White
Write-Host "  Log        : $logFile" -ForegroundColor DarkGray
Write-Host "  Dashboard  : http://localhost:$Port" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# ─── Launch translator (background) ──────────────────────────────────────
$translatorArgs = @("main.py", "--system-audio", "--log", $logFile)
if ($Streaming) { $translatorArgs += "--streaming" }

try {
    $translatorJob = Start-Process -FilePath "python" -ArgumentList $translatorArgs `
        -NoNewWindow -PassThru
} catch {
    Write-Host "Failed to start translator: $_" -ForegroundColor Red
    exit 1
}

Write-Host "  [PID $($translatorJob.Id)] Translator started" -ForegroundColor Green
Start-Sleep -Seconds 3

# ─── Launch Streamlit (foreground) ────────────────────────────────────────
try {
    streamlit run webui\streamlit_app.py --server.port $Port
} finally {
    if (-not $translatorJob.HasExited) {
        Write-Host "`n  Stopping translator (PID $($translatorJob.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $translatorJob.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  Done. Log: $logFile" -ForegroundColor Green
}
