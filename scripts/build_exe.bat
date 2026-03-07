@echo off
REM Build ScholarGenie CLI Executable

echo ===============================================
echo   Building ScholarGenie CLI Executable
echo ===============================================
echo.

cd ..
call venv\Scripts\activate

echo [1/3] Installing build dependencies...
pip install pyinstaller rich click pyfiglet

echo.
echo [2/3] Building executable with PyInstaller...
pyinstaller --onefile ^
    --name ScholarGenie ^
    --icon scripts\cli\icon.ico ^
    --add-data "backend;backend" ^
    --hidden-import=backend.agents.paper_finder ^
    --hidden-import=backend.agents.pdf_parser ^
    --hidden-import=backend.agents.summarizer ^
    --hidden-import=backend.agents.presenter ^
    --hidden-import=rich ^
    --hidden-import=click ^
    --noconsole=False ^
    scripts\cli\scholargenie.py

echo.
echo [3/3] Cleaning up...
move dist\ScholarGenie.exe .
rmdir /s /q build dist
del ScholarGenie.spec

echo.
echo ===============================================
echo   Build Complete!
echo ===============================================
echo.
echo   Executable: ScholarGenie.exe
echo   Size: ~50MB (includes all dependencies)
echo.
echo   Usage:
echo     ScholarGenie.exe              (Interactive mode)
echo     ScholarGenie.exe search "quantum computing"
echo     ScholarGenie.exe version
echo.
pause
