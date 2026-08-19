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

if not exist ".venv" (
    echo [warn] .venv not found - run "uv sync" first.
)
if not exist "frontend\node_modules" (
    echo [warn] frontend\node_modules not found - run "npm install" in frontend\ first.
)

echo Starting infra (postgres, redis, milvus)...
docker compose up -d

echo Applying sis database migrations...
call .venv\Scripts\alembic.exe -c sis\alembic.ini upgrade head

echo Starting identity :8200 ...
start "identity :8200" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && set IDENTITY_ADMIN_KEY=dev-admin-key && uvicorn identity.app:app --port 8200"

echo Starting records :8100 ...
start "records :8100" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && set RECORDS_BOOTSTRAP_ADMIN_KEY=dev-records-admin && set IDENTITY_JWKS_URL=http://localhost:8200/.well-known/jwks.json && uvicorn records.app:app --port 8100"

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
