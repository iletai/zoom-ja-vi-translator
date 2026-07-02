@echo off
:: Double-click this file to start NLLB translator + web dashboard
:: (Bypasses PowerShell execution policy automatically)
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run-nllb.ps1"
pause
