# Chạy translator với Qwen2.5-1.5B LLM (chất lượng tốt nhất cho meeting IT)
# Không cần tham số — chỉ cần chạy: .\run-qwen.ps1
#
# Qwen2.5-1.5B: dịch có ngữ cảnh, hiểu thuật ngữ IT tốt hơn NLLB.
# RAM: ~1.1GB (model) + ~160MB (ASR) = ~1.3GB tổng cộng.

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
    python -m pip install "llama-cpp-python>=0.2.90,<0.3.0" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -q
    python -m pip install numpy soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm pip-system-certs certifi -q
    New-Item -ItemType File -Path ".venv\.deps_installed" | Out-Null
}

# ─── Models ───────────────────────────────────────────────────────────────
if (-not (Test-Path "models\reazonspeech-k2-v2")) {
    Write-Host "==> Downloading ASR model (~160 MB)..." -ForegroundColor Yellow
    python scripts\download_models.py
}

$model1p5b = Join-Path $PSScriptRoot "models\qwen2.5-1.5b-instruct\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
$model3b = Join-Path $PSScriptRoot "models\qwen2.5-3b-instruct\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
if (-not (Test-Path $model1p5b) -and -not (Test-Path $model3b)) {
    Write-Host "==> Downloading Qwen2.5-1.5B (~0.9 GB)..." -ForegroundColor Yellow
    python scripts\download_qwen_model.py --size 1.5b
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Download failed!" -ForegroundColor Red
        exit 1
    }
}

# ─── Environment ──────────────────────────────────────────────────────────
$env:ZT_TRANSLATOR = "llm"
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
$modelName = if (Test-Path $model1p5b) { "Qwen2.5-1.5B" } else { "Qwen2.5-3B" }
Write-Host ""
Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║  Zoom JA→VI Translator [Qwen LLM]    ║" -ForegroundColor Cyan
Write-Host "  ╚═══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Model     : $modelName (Q4_K_M, context-aware)" -ForegroundColor White
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
