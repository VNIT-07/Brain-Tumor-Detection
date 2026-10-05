@echo off
REM ==============================================================================
REM NeuroScan AI - Launcher Script (Windows)
REM ==============================================================================

echo ============================================================
echo   NeuroScan AI - Brain Tumor Detection Interface
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9 - 3.11 from python.org and check "Add to PATH".
    pause
    exit /b 1
)

REM Setup Virtual Environment if missing
if not exist ".venv" (
    echo [SETUP] Creating virtual environment (.venv)...
    python -m venv .venv
    echo [SETUP] Installing dependencies...
    .venv\Scripts\pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt
)

REM Ensure local Ultralytics and Streamlit directories
if not exist ".ultralytics" mkdir .ultralytics
set YOLO_CONFIG_DIR=%cd%\.ultralytics
set STREAMLIT_CREDENTIALS_FILE=%cd%\.streamlit\credentials.toml
set STREAMLIT_CONFIG_FILE=%cd%\.streamlit\config.toml

echo [LAUNCH] Starting NeuroScan AI...
.venv\Scripts\streamlit run app.py

pause
