@echo off
setlocal

cd /d "%~dp0"

set "APP_ROOT=%CD%"
set "VENV_PY=%APP_ROOT%\.venv\Scripts\python.exe"
set "SRC_DIR=%APP_ROOT%\src"
set "MAIN_MODULE=untype.main"

if /i "%UNTYPE_DEV_DRY_RUN%"=="1" (
    if not exist "%VENV_PY%" (
        echo [DRY RUN] uv sync --extra dev --extra test
    )
    echo [DRY RUN] "%VENV_PY%" -m %MAIN_MODULE%
    exit /b 0
)

if not exist "%VENV_PY%" (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Missing Python runtime: "%VENV_PY%"
        echo [ERROR] uv was not found on PATH, so the dev environment could not be created.
        echo.
        echo Install uv or create the virtualenv manually, then try again.
        echo.
        pause
        exit /b 1
    )

    echo [SETUP] Dev environment not found. Bootstrapping with uv...
    call uv sync --extra dev --extra test
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create the dev environment.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%VENV_PY%" (
    echo [ERROR] Python runtime still missing after setup: "%VENV_PY%"
    echo.
    pause
    exit /b 1
)

if not exist "%SRC_DIR%\untype\main.py" (
    echo [ERROR] Missing source entry point: "%SRC_DIR%\untype\main.py"
    echo.
    pause
    exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%SRC_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%SRC_DIR%"
)
set "PYTHONUTF8=1"

echo Launching UnType in dev mode...
echo Root   : "%APP_ROOT%"
echo Python : "%VENV_PY%"
echo.

start "UnType Dev" "%VENV_PY%" -m %MAIN_MODULE%
if errorlevel 1 (
    echo [ERROR] Failed to start UnType.
    echo.
    pause
    exit /b 1
)

echo UnType dev launch request sent.
echo Check the system tray for the green icon.
exit /b 0
