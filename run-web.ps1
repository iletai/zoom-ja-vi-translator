# Launch the live Streamlit dashboard for the Zoom JA->VI translator (Windows / PowerShell).
#
# The dashboard only *tails* the JSONL evidence log the translator writes, so it
# can never add latency or drop data. Start the translator separately with
# logging, e.g.:
#   ./run.ps1 -SystemAudio --Log test_audio\evidence\live.jsonl
# then pick that file in the dashboard sidebar.
#
# Usage:
#   ./run-web.ps1                       # launch dashboard on http://localhost:8501
#   ./run-web.ps1 --server.port 8600    # extra args pass through to streamlit

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

# Install the web dependency once (separate marker from the main ML deps).
if (-not (Test-Path ".venv\.web_deps_installed")) {
    Write-Host "==> Installing web dependencies (streamlit)..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements-web.txt
    New-Item -ItemType File -Path ".venv\.web_deps_installed" | Out-Null
}

Write-Host "==> Starting Streamlit dashboard -> http://localhost:8501"
streamlit run webui\streamlit_app.py @args
