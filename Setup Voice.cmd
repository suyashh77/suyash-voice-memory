@echo off
setlocal
cd /d "%~dp0"
if not exist "temp" mkdir "temp"
set "TEMP=%~dp0temp"
set "TMP=%~dp0temp"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" setup_voice.py
if errorlevel 1 goto failed
echo.
echo Ready. Double-click Start Voice.cmd.
pause
exit /b 0
:failed
echo.
echo Setup could not finish. The message above explains why.
pause
exit /b 1
