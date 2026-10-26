@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" connect_claude.py
pause
