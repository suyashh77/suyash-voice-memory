@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" setup_voice.py --model large-v3-turbo
echo Select the downloaded model in Recognition settings, then restart the app.
pause
