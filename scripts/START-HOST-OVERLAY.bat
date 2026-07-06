@echo off
:: Double-click to start web UI host with transparent subtitle overlay
:: Uses 9router backend + auto-start on browser connect
:: Requires ZT_HOST_REAL=1 for real audio capture (set in run-host.ps1)
:: Bypasses PowerShell execution policy automatically
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run-host.ps1" -Overlay -AutoStart
pause
