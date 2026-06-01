# Chạy translator với NLLB-600M (nhanh, nhẹ RAM, không cần LLM)
# Không cần tham số — chỉ cần chạy: .\run-nllb.ps1
#
# NLLB-600M: dịch nhanh nhưng không có ngữ cảnh giữa các câu.
# RAM: ~600MB (model) + ~160MB (ASR) = ~760MB tổng cộng.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ─── UTF-8 ────────────────────────────────────────────────────────────────
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# ─── Venv ─────────────────────────────────────────────────────────────────
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) { $python = "py -3.11" }
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

# ─── Dependencies ─────────────────────────────────────────────────────────
if (-not (Test-Path ".venv\.deps_installed")) {
    Write-Host "==> Installing dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip -q
    python -m pip install webrtcvad-wheels -q
    python -m pip install "ctranslate2==4.5.0" "setuptools<70" -q
    python -m pip install numpy soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm pip-system-certs certifi -q
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}

# ─── Models ───────────────────────────────────────────────────────────────
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading ASR model (~160 MB)..." -ForegroundColor Yellow
    python scripts\download_models.py
}

# ─── Environment ──────────────────────────────────────────────────────────
$env:ZT_TRANSLATOR = "nllb"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# ─── Web dependencies ─────────────────────────────────────────────────────
if (-not (Test-Path ".venv\.web_deps_installed")) {
    Write-Host "==> Installing web dependencies (streamlit)..." -ForegroundColor Yellow
    python -m pip install -r requirements-web.txt -q
    New-Item -ItemType File -Path ".venv\.web_deps_installed" | Out-Null
}

# ─── Log file ─────────────────────────────────────────────────────────────
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PSScriptRoot "test_audio\evidence"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir "run_${timestamp}.jsonl"

# ─── Banner ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║  Zoom JA→VI Translator [NLLB]        ║" -ForegroundColor Cyan
Write-Host "  ╚═══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Model     : NLLB-600M (CTranslate2, int8)" -ForegroundColor White
Write-Host "  ASR       : ReazonSpeech k2-v2 + IT hotwords" -ForegroundColor White
Write-Host "  Audio     : System audio (WASAPI loopback)" -ForegroundColor White
Write-Host "  Log       : $logFile" -ForegroundColor DarkGray
Write-Host "  Dashboard : http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# ─── Launch translator (background) ──────────────────────────────────────
$translatorJob = Start-Process -FilePath "python" `
    -ArgumentList @("main.py", "--system-audio", "--log", $logFile) `
    -NoNewWindow -PassThru

Write-Host "  [PID $($translatorJob.Id)] Translator started" -ForegroundColor Green
Start-Sleep -Seconds 3

# ─── Launch Streamlit (foreground) ────────────────────────────────────────
try {
    streamlit run webui\streamlit_app.py --server.port 8501
} finally {
    if (-not $translatorJob.HasExited) {
        Write-Host "`n  Stopping translator (PID $($translatorJob.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $translatorJob.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  Done. Log: $logFile" -ForegroundColor Green
}
