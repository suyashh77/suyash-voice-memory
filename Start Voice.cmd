@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First run Setup Voice.cmd while connected to the internet.
  pause
  exit /b 1
)
start "Suyash Voice" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
