@echo off
title ScholarGenie
color 0A

echo.
echo  ==========================================
echo   ScholarGenie — AI Research Platform
echo  ==========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] venv not found. Run: python -m venv venv
    pause & exit /b 1
)

REM Check Node.js
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause & exit /b 1
)

REM Install frontend dependencies if needed
if not exist frontend\node_modules (
    echo [SETUP] Installing frontend packages...
    cd frontend && npm install --silent && cd ..
)

echo  Starting API server on port 8000...
start "ScholarGenie API" cmd /k "title ScholarGenie API && python api.py"

echo  Waiting for API to start...
timeout /t 4 /nobreak >nul

echo  Starting Frontend on port 3000...
start "ScholarGenie Frontend" cmd /k "title ScholarGenie Frontend && cd frontend && npm run dev"

echo  Waiting for frontend to compile...
timeout /t 8 /nobreak >nul

echo.
echo  ==========================================
echo   ScholarGenie is running!
echo.
echo   App:      http://localhost:3000
echo   API Docs: http://localhost:8000/docs
echo  ==========================================
echo.

start http://localhost:3000
exit
