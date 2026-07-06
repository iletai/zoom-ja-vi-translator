@echo off
:: Double-click to start the translator with transparent subtitle overlay
:: Uses 9router backend (requires 9router gateway running locally)
:: Bypasses PowerShell execution policy automatically
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run.ps1" -Router -Overlay
pause
