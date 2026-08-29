@echo off
setlocal
cd /d "%~dp0"
title SuperAgent Docker Manager
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0devops\superagent.ps1" %*
if errorlevel 1 (
  echo.
  echo SuperAgent manager ended with an error.
  pause
)
