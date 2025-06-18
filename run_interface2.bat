@echo off
setlocal

REM Set venv path and Python script name
set VENV_DIR=venv
set SCRIPT_NAME=interface2.py

REM Check if venv exists
if not exist %VENV_DIR%\Scripts\activate.bat (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv %VENV_DIR%

    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo [INFO] Virtual environment created.
)

REM Activate the venv
call %VENV_DIR%\Scripts\activate.bat

REM Install requirements if requirements.txt exists
if exist requirements.txt (
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
)

REM Run the script
echo [INFO] Running %SCRIPT_NAME%...
python %SCRIPT_NAME%

pause
endlocal
