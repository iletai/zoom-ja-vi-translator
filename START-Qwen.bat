@echo off
:: Double-click this file to start Qwen2.5 translator + web dashboard
:: (Bypasses PowerShell execution policy automatically)
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0run-qwen.ps1"
pause
