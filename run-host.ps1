# Launch the Speaksy web UI host with 9router translation (Windows / PowerShell).
#
# Serves the React UI (webui\rd_ui_v1.1.html) and bridges it to the Python
# pipeline over WebSocket, translating Japanese to Vietnamese through the local
# 9router gateway instead of a heavy in-process LLM (Qwen). ASR still runs
# locally (it must, to hear the meeting), but the slow part -- translation -- is
# offloaded to 9router, so no Qwen/llama-cpp model is downloaded or loaded.
#
# Modes:
#   ./run-host.ps1            # REAL audio: capture system audio -> ASR -> 9router
#   ./run-host.ps1 -Mic       # REAL audio from the microphone instead of loopback
#   ./run-host.ps1 -Demo      # no audio: scripted Japanese, translated live by 9router
#   ./run-host.ps1 -ListDevices
#
# Options:
#   -Port 8770                # HTTP/WebSocket port (default 8770)
#   -Model "gh/claude-haiku-4.5"   # 9router model id (see GET /v1/models)
#
# After it starts, open the printed http://127.0.0.1:<port> URL in a browser
# and click the start button.

param(
    [switch]$Demo,
    [switch]$Mic,
    [switch]$ListDevices,
    [int]$Port = 8770,
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# --- UTF-8 console (Japanese / Vietnamese output) -------------------------
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

# --- Python interpreter ---------------------------------------------------
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    $python = "py -3.11"
}

# --- Virtual environment (shared with run.ps1) ----------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtual environment..." -ForegroundColor Yellow
    & $python -m venv .venv
}
& ".venv\Scripts\Activate.ps1"

# --- SSL fix (corporate proxies / missing CA certs) -----------------------
$certifi = python -c "import certifi; print(certifi.where())" 2>$null
if ($certifi) {
    $env:SSL_CERT_FILE = $certifi
    $env:REQUESTS_CA_BUNDLE = $certifi
}

# --- Dependencies ---------------------------------------------------------
# The translator only needs 'requests' (it just POSTs to 9router). REAL audio
# additionally needs the ASR stack -- but NOT llama-cpp/Qwen, which is the whole
# point: we skip the heavy local translation model.
python -c "import requests" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Installing translator dependency (requests)..." -ForegroundColor Yellow
    python -m pip install requests certifi pip-system-certs -q
}

if (-not $Demo) {
    if (-not (Test-Path ".venv\.host_real_deps_installed")) {
        Write-Host "==> Installing ASR dependencies (no Qwen/llama-cpp)..." -ForegroundColor Yellow
        python -m pip install --upgrade pip -q
        python -m pip install webrtcvad-wheels -q
        python -m pip install "ctranslate2==4.5.0" "setuptools<70" -q
        python -m pip install numpy scipy soxr soundcard sherpa-onnx transformers sentencepiece huggingface_hub tqdm -q
        New-Item -ItemType File -Path ".venv\.host_real_deps_installed" | Out-Null
    }

    # ASR model (~2.5 GB, first run only). No Qwen model is ever downloaded.
    if (-not (Test-Path "models\reazonspeech-k2-v2")) {
        Write-Host "==> Downloading ASR models (~2.5 GB, first run only)..." -ForegroundColor Yellow
        python scripts\download_models.py
    }

    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:ZT_HOST_REAL = "1"
}

# --- Translator backend: 9router ------------------------------------------
$env:ZT_TRANSLATOR = "router"
if ($Model -ne "") { $env:ZT_ROUTER_MODEL = $Model }
$routerModel = if ($Model -ne "") { $Model } else { "gh/claude-haiku-4.5 (default)" }
$routerBase = if ($env:ZT_ROUTER_BASE_URL) { $env:ZT_ROUTER_BASE_URL } else { "http://127.0.0.1:20128/v1" }

# --- Probe the gateway so failures are obvious up front -------------------
$routerKey = if ($env:ZT_ROUTER_KEY) { $env:ZT_ROUTER_KEY } else { "sk_9router" }
$probe = "import requests,sys; sys.exit(0 if requests.get('$routerBase/models', headers={'Authorization':'Bearer $routerKey'}, timeout=3).ok else 1)"
try { python -c $probe 2>$null } catch {}
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [warn] 9router not reachable at $routerBase - start it first, or the host falls back to scripted text." -ForegroundColor Red
} else {
    Write-Host "  [ok] 9router reachable at $routerBase" -ForegroundColor Green
}

# --- List devices and exit ------------------------------------------------
if ($ListDevices) {
    python main.py --list-devices
    return
}

# --- Show config ----------------------------------------------------------
$audioMode = if ($Demo) { "Demo (scripted Japanese, no audio)" }
             elseif ($Mic) { "Microphone" }
             else { "System audio (WASAPI loopback)" }
$asrMode = if ($Demo) { "(disabled in demo)" } else { "ReazonSpeech k2-v2 (local)" }
Write-Host ""
Write-Host "  Speaksy Web UI Host" -ForegroundColor Cyan
Write-Host "  -------------------" -ForegroundColor DarkGray
Write-Host "  Translator : 9router -> $routerModel" -ForegroundColor White
Write-Host "  ASR        : $asrMode" -ForegroundColor White
Write-Host "  Audio      : $audioMode" -ForegroundColor White
Write-Host "  UI         : http://127.0.0.1:$Port" -ForegroundColor White
Write-Host ""
Write-Host "  Open the URL above in a browser, then click the start button." -ForegroundColor DarkGray
Write-Host ""

# --- Run the host ---------------------------------------------------------
# -Mic flips the real engine from loopback to the default microphone.
if ($Mic) { $env:ZT_HOST_MIC = "1" }
python webui\host_server.py --host 127.0.0.1 --port $Port @args
