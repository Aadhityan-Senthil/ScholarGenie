@echo off
echo ============================================
echo   ScholarGenie CLI - Quick Start
echo ============================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Check if dependencies are installed
python -c "import rich" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_final.txt
)

REM Run CLI
python scholargenie_v2.py

pause
