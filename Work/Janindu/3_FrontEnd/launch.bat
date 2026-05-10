@echo off
echo ============================================================
echo   Sinhala OCR Pipeline - Frontend Launcher
echo ============================================================

cd /d "%~dp0"

REM ── Try venv python first ─────────────────────────────────────────────────
set PYTHON="..\2_Recogniton\venv311\Scripts\python.exe"
if not exist %PYTHON% (
    echo [WARN] venv311 python not found, using system python
    set PYTHON=python
)

REM ── Install Flask if needed ───────────────────────────────────────────────
%PYTHON% -c "import flask" 2>nul
if errorlevel 1 (
    echo [INFO] Installing Flask...
    %PYTHON% -m pip install flask --quiet
)

echo.
echo [INFO] Starting server at http://localhost:5000
echo [INFO] Press Ctrl+C to stop.
echo.

REM ── Open browser after short delay ───────────────────────────────────────
start "" /B cmd /c "timeout /t 2 >nul && start http://localhost:5000"

%PYTHON% app.py

pause
