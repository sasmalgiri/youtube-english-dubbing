@echo off
chcp 65001 >nul 2>&1
title VoiceDub

:: ── Find Python ──
set PYTHON=

:: Check common locations
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
    goto :found_python
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
    goto :found_python
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
    goto :found_python
)
if exist "C:\Python310\python.exe" (
    set PYTHON=C:\Python310\python.exe
    goto :found_python
)
if exist "C:\Python311\python.exe" (
    set PYTHON=C:\Python311\python.exe
    goto :found_python
)

:: Check PATH
where python >nul 2>&1
if %errorlevel%==0 (
    :: Verify it actually works (not a broken shim)
    python --version >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
        goto :found_python
    )
)

:: Check py launcher
where py >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py -3
    goto :found_python
)

echo.
echo   [ERROR] Python 3.10+ not found!
echo.
echo   Install Python from: https://www.python.org/downloads/
echo   Or run: winget install Python.Python.3.10
echo.
echo   IMPORTANT: Check "Add Python to PATH" during installation!
echo.
pause
exit /b 1

:found_python
set PYTHONIOENCODING=utf-8
set COQUI_TOS_AGREED=1
cd /d "%~dp0"
%PYTHON% desktop.py
if %errorlevel% neq 0 (
    echo.
    echo   Something went wrong. Check the error above.
    pause
)
