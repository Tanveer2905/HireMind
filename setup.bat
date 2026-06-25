@echo off
REM ============================================================
REM  AI Hiring Copilot — Setup Script (Windows)
REM  Creates venv, installs dependencies, downloads models.
REM  Run this once before first use.
REM ============================================================

echo.
echo ============================================================
echo   AI HIRING COPILOT - SETUP
echo   This will set up the complete environment.
echo   Estimated time: 10-20 minutes (depending on internet speed)
echo ============================================================
echo.

REM --- Navigate to project root ---
cd /d "%~dp0"

REM --- Set local cache directories to keep C drive clean ---
set HF_HOME=%~dp0data\.cache\huggingface

REM --- Create virtual environment ---
echo [1/5] Creating virtual environment...
if exist "venv" (
    echo   Virtual environment already exists.
    echo   Delete the 'venv' folder to recreate.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo   ERROR: Failed to create virtual environment.
        echo   Make sure Python 3.8+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo   Virtual environment created.
)
echo.

REM --- Activate virtual environment ---
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo   ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo   Activated.
echo.

REM --- Install dependencies ---
echo [3/5] Installing dependencies...
echo   This may take several minutes...
echo.
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo   ERROR: Failed to install dependencies.
    echo   Check your internet connection and try again.
    pause
    exit /b 1
)
echo.
echo   Installing pre-compiled llama-cpp-python...
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
echo.
echo   Dependencies installed.
echo.

REM --- Download models ---
echo [4/4] Downloading and saving models locally...
echo   This may take a while (~5GB total)...
echo.
python download_models.py
if errorlevel 1 (
    echo   ERROR: Failed to download models.
    echo   Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

REM --- Create required directories ---
if not exist "resumes" mkdir resumes
if not exist "data" mkdir data

REM --- Done ---
echo.
echo ============================================================
echo   SETUP COMPLETE!
echo.
echo   Next steps:
echo   1. Place resume PDFs in the 'resumes' folder
echo   2. Run 'run.bat' to start the system
echo.
echo   The core system works fully OFFLINE and all models are stored
echo   directly inside this project folder.
echo ============================================================
echo.
pause
