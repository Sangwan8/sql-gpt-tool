@echo off
echo ==================================================
echo NLQ to SQL Query Generator - Startup Script
echo ==================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/Update dependencies
echo.
echo Checking dependencies...
pip install -q -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create .env from .env.example and add your OpenAI API key
    echo.
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs

REM Start the application
echo.
echo ==================================================
echo Starting NLQ to SQL Generator...
echo ==================================================
echo.
echo Application will be available at: http://localhost:5001
echo Press Ctrl+C to stop the server
echo.
echo ==================================================
echo.

python nlq_sql_app.py

pause
