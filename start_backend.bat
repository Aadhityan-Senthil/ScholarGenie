@echo off
echo Starting ScholarGenie Backend...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Check if running in Docker or local
echo Choose startup mode:
echo [1] Docker (Recommended - includes GROBID)
echo [2] Local (Python only - no PDF parsing)
echo.

set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting with Docker Compose...
    echo This will start:
    echo   - GROBID PDF Parser (port 8070)
    echo   - FastAPI Backend (port 8000)
    echo   - Streamlit Demo (port 8501)
    echo.
    docker-compose up -d
    echo.
    echo Services started! Access at:
    echo   - API: http://localhost:8000
    echo   - Demo UI: http://localhost:8501
    echo   - API Docs: http://localhost:8000/docs
) else (
    echo.
    echo Starting FastAPI locally...
    echo Note: PDF parsing won't work without GROBID
    echo.
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
)

pause
