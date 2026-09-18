@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error. Please verify your environment and dependencies.
    pause
)
