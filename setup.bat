@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Turbo CLI Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from python.org
    pause
    exit /b 1
)

echo Step 1: Bundling engine...
python package_engine.py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to bundle engine
    pause
    exit /b 1
)

echo Step 2: Installing...
python -m pip install -e . >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Installation failed
    pause
    exit /b 1
)

echo Step 3: Verifying...
turbo --help >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Run 'turbo launch' to start the server.
echo.
pause
