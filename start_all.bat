@echo off
REM =========================================================================
REM  start_all.bat — Start the full Sinhala Reader system (all 4 components)
REM
REM  USAGE:
REM    start_all.bat                    (start everything)
REM    start_all.bat --no-layout        (skip PaddleOCR layout detection)
REM
REM  WHAT THIS STARTS:
REM    Terminal 1: Voice Service     (Component 4, Bumal)    port 8101
REM    Terminal 2: RAG Service       (Component 3, Nadee)    port 8102
REM    Terminal 3: Main Pipeline     (Components 1+2, Ishara) port 8000
REM
REM  WHAT'S ALREADY ENABLED VIA .env:
REM    SINHALA_TITLE_MODE=mat         (Component 1: title OCR)
REM    SINHALA_VOICE_MODE=http        (Component 4: voice → port 8101)
REM    SINHALA_RAG_MODE=http          (Component 3: RAG   → port 8102)
REM
REM  TO CONNECT THE PHONE:
REM    adb reverse tcp:8000 tcp:8000
REM =========================================================================

set REPO_ROOT=%~dp0
cd /d "%REPO_ROOT%"

echo.
echo  ╔═══════════════════════════════════════════════╗
echo  ║    Sinhala Reader — Full System Startup       ║
echo  ║    R26-DS-002 · All 4 Components              ║
echo  ╚═══════════════════════════════════════════════╝
echo.

REM --- Free ports 8000, 8101, 8102 if held by previous runs ---
powershell -Command "Get-NetTCPConnection -LocalPort 8000, 8101, 8102 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" > nul 2>&1
timeout /t 1 /nobreak > nul

REM --- 1. Start Voice Service (Component 4) ---
echo [1/3] Starting Voice Service (Component 4 — Bumal) on port 8101...
start "SVC-VOICE (Port 8101)" cmd /k "cd /d %REPO_ROOT% && python services\voice\app.py --port 8101"
timeout /t 3 /nobreak > nul

REM --- 2. Start RAG Service (Component 3) ---
echo [2/3] Starting RAG Service (Component 3 — Nadee) on port 8102...
start "SVC-RAG (Port 8102)" cmd /k "cd /d %REPO_ROOT% && python services\rag\app.py --port 8102"
timeout /t 3 /nobreak > nul

REM --- 3. Start Main Pipeline Server ---
echo [3/3] Starting Main Pipeline Server on port 8000...
echo.
echo  All services starting. The main server will load YOLO and mT5 models.
echo  This may take 30-60 seconds on first run.
echo.
echo  Once ready you will see:
echo    phone should POST to  http://^<this-machine^>:8000/capture
echo    browser test page      http://127.0.0.1:8000/
echo.
echo  To connect the phone:
echo    adb reverse tcp:8000 tcp:8000
echo.

cd /d "%REPO_ROOT%\system"
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2 %*
