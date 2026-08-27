@echo off
REM One-click local dev launcher. Uses the existing .venv directly (no
REM dependency on "uv" being on PATH). Assumes one-time setup is already done:
REM   uv sync   (or: python -m venv .venv && pip install -e .)
REM   .venv\Scripts\pip install -r identity/requirements.txt -r records/requirements.txt -r sis/requirements.txt
REM   copy .env.example .env   (edited)
REM   cd sis && ..\.venv\Scripts\alembic upgrade head
REM   cd frontend && npm install
REM
REM Opens one window per service. Close a window (or Ctrl+C in it) to stop that
REM service; closing this launcher window does not stop the others.

cd /d "%~dp0"

REM Needed for !BUSY! below: a variable built inside a for-loop is
REM only readable with delayed expansion.
setlocal enabledelayedexpansion

if not exist ".venv" (
    echo [warn] .venv not found - run "uv sync" first.
)
if not exist "frontend\node_modules" (
    echo [warn] frontend\node_modules not found - run "npm install" in frontend\ first.
)

REM Every service now reads .env for itself (identity/env.py, records/env.py,
REM sis/env.py, backend/env.py). Nothing is set here any more, deliberately: a variable
REM exported in this window BEATS the file, so the keys that used to be pinned here won
REM silently over whatever .env said — which is how records came to hold a bootstrap
REM admin key that nobody had written down and nobody could use. That key is gone with
REM the table behind it; records now has one credential, RECORDS_API_KEY, and it is the
REM same value the chat backend sends.
REM
REM Configure the estate in .env. This file only starts processes.

REM A service reads .env once, at startup. If an older copy still holds the port, the
REM new one cannot bind, exits into its own window, and the OLD one keeps answering with
REM the OLD settings - which looks exactly like an edit to .env not working.
REM
REM So this stops rather than starting half an estate. It offers to free the ports
REM because the alternative is a netstat/taskkill line nobody should have to retype.

set "BUSY="
for %%P in (8000 8100 8200 8300) do (
    netstat -ano -p TCP | findstr /R /C:"LISTENING" | findstr /C:":%%P " >nul && set "BUSY=!BUSY! %%P"
)

if defined BUSY (
    echo.
    echo [busy] These ports are already in use:!BUSY!
    echo        A previous run is still going. Starting now would leave the OLD services
    echo        answering with the OLD .env - which reads exactly like your edit being
    echo        ignored.
    echo.
    choice /c YN /m "Stop them and start fresh"
    if errorlevel 2 (
        echo Left running. Nothing was started.
        pause
        exit /b 1
    )
    for %%P in (8000 8100 8200 8300) do (
        for /f "tokens=5" %%i in ('netstat -ano -p TCP ^| findstr /C:":%%P " ^| findstr LISTENING') do (
            taskkill /PID %%i /F >nul 2>&1
        )
    )
    echo Stopped. Their windows will show the shutdown; you can close them.
    timeout /t 2 >nul
)

echo Starting infra (postgres, redis, milvus)...
docker compose up -d

echo Applying sis database migrations...
call .venv\Scripts\alembic.exe -c sis\alembic.ini upgrade head

echo Starting identity :8200 ...
start "identity :8200" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && uvicorn identity.app:app --port 8200"

echo Starting records :8100 ...
start "records :8100" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && uvicorn records.app:app --port 8100"

echo Starting sis :8300 (UI at /ui, no API key required) ...
start "sis :8300" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && uvicorn sis.app:app --port 8300"

echo Starting chat backend :8000 ...
start "backend :8000" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

echo Starting chat frontend dev server :3000 ...
start "frontend :3000" cmd /k "cd /d "%~dp0\frontend" && npm run dev"

echo.
echo All services launching in separate windows:
echo   Chat UI      http://localhost:3000
echo   Chat backend http://localhost:8000/docs
echo   identity     http://localhost:8200/docs
echo   records      http://localhost:8100/docs
echo   sis + UI     http://localhost:8300/ui   (http://localhost:8300/docs)
echo.
pause
