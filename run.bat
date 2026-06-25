@echo off
REM ============================================================
REM  AI Recruiter — Run Script (Windows)
REM  Activates the virtual environment and runs the system.
REM ============================================================

REM --- Navigate to project root ---
cd /d "%~dp0"

REM --- Set local cache directories to keep C drive clean ---
set HF_HOME=%~dp0data\.cache\huggingface
set TEMP=%~dp0data\temp
set TMP=%~dp0data\temp

REM Ensure the temp directory exists
if not exist "%~dp0data\temp" mkdir "%~dp0data\temp"

REM --- Check if venv exists ---
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo   ERROR: Virtual environment not found.
    echo   Please run 'setup.bat' first.
    echo.
    pause
    exit /b 1
)

REM --- Check for resumes ---
dir /b "resumes\*.pdf" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   WARNING: No PDF files found in the 'resumes' folder.
    echo   Place resume PDFs there before running.
    echo.
)

REM --- Activate and run ---
call venv\Scripts\activate.bat
python -m backend.main %*

REM --- Keep window open ---
echo.
pause
