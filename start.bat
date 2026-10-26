@echo off
setlocal
cd /d "%~dp0"
title AppScout

rem Find Python: the "py" launcher first, then python on the PATH.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
    echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then run start.bat again.
    pause
    exit /b 1
)

rem First run only: a private Python environment with the packages AppScout needs.
if not exist ".venv\setup-done.txt" (
    echo Setting up AppScout for the first time. This takes a few minutes...
    if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto :setup_failed
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :setup_failed
    echo done> ".venv\setup-done.txt"
)

echo.
echo Starting AppScout. Your browser opens on it in a few seconds; the first load reads 9,659 apps
echo and takes about 20 seconds. Close this window to stop AppScout.
rem No fixed port: if 8501 is taken (AppScout already open), Streamlit uses the next free one and opens
rem the browser there. The empty line answers Streamlit's one-time email question so it never waits.
set "HEADLESS=false"
if defined APPSCOUT_NO_BROWSER set "HEADLESS=true"
echo.| ".venv\Scripts\python.exe" -m streamlit run app\app.py --server.headless %HEADLESS% --browser.gatherUsageStats false
pause
exit /b 0

:setup_failed
echo.
echo Setting up failed. Check your internet connection, then run start.bat again.
if exist .venv rmdir /s /q .venv
pause
exit /b 1
