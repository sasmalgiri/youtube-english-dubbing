@echo off
chcp 65001 >nul 2>&1
title VoiceDub - Setup
color 0E

echo.
echo   ================================================
echo    VoiceDub - First Time Setup
echo   ================================================
echo.

:: ── Find Python ──
set PYTHON=
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
    ) else (
        echo   [ERROR] Python 3.10+ not found!
        echo   Install from: https://python.org/downloads
        echo   Or run: winget install Python.Python.3.10
        pause
        exit /b 1
    )
)
echo   [OK] Python: %PYTHON%
%PYTHON% --version

:: ── Check Node.js ──
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   Node.js not found. Installing...
    winget install OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
    if %errorlevel% neq 0 (
        echo   [ERROR] Node.js install failed. Install manually from https://nodejs.org
        pause
        exit /b 1
    )
    echo   [NOTE] Close and reopen this terminal, then run setup.bat again.
    pause
    exit /b 0
) else (
    echo   [OK] Node.js found
    node --version
)

:: ── Check FFmpeg ──
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   FFmpeg not found. Installing...
    winget install Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    echo   [NOTE] FFmpeg installed. Restart terminal for PATH changes.
) else (
    echo   [OK] FFmpeg found
)

:: ── Install Python packages ──
echo.
echo   Installing Python packages...
cd /d "%~dp0backend"
%PYTHON% -m pip install --upgrade pip --quiet
%PYTHON% -m pip install -r requirements.txt --quiet
echo   [OK] Core packages installed

:: ── GPU Setup ──
echo.
%PYTHON% -c "import torch; cuda=torch.cuda.is_available(); print(f'  GPU: {torch.cuda.get_device_name(0)}' if cuda else '  GPU: Not available (CPU mode)')" 2>nul
if %errorlevel% neq 0 (
    echo   No PyTorch detected.
)

echo.
set /p INSTALL_GPU="  Install GPU packages (Coqui XTTS, Chatterbox AI)? Needs NVIDIA GPU. (y/N): "
if /i "%INSTALL_GPU%"=="y" (
    echo.
    echo   Installing PyTorch with CUDA 12.6 support...
    %PYTHON% -m pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    echo.
    echo   Installing Coqui XTTS v2...
    %PYTHON% -m pip install TTS
    echo.
    echo   Installing Chatterbox AI...
    %PYTHON% -m pip install chatterbox-tts
    echo.
    echo   Installing speaker diarization...
    %PYTHON% -m pip install pyannote-audio
    echo.
    echo   Re-installing PyTorch CUDA (in case a package downgraded it^)...
    %PYTHON% -m pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    echo.
    %PYTHON% -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    echo   [OK] GPU packages installed
)

:: ── Install frontend packages ──
echo.
echo   Installing frontend packages...
cd /d "%~dp0web"
call npm install --quiet 2>nul
echo   [OK] Frontend packages installed

:: ── Create .env ──
cd /d "%~dp0backend"
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo   [OK] Created .env from .env.example
    ) else (
        echo # Add your API keys here> .env
        echo   [OK] Created empty .env
    )
    echo.
    echo   IMPORTANT: Edit backend\.env and add your API keys!
    echo   At minimum, add one translation API key:
    echo     GEMINI_API_KEY=your_key   (free tier at aistudio.google.com)
    echo     OPENAI_API_KEY=your_key   (paid, best quality)
    echo     GROQ_API_KEY=your_key     (free, fast)
) else (
    echo   [OK] .env already exists
)

echo.
echo   ================================================
echo    Setup complete!
echo.
echo    1. Edit backend\.env with your API keys
echo    2. Double-click VoiceDub.bat to launch
echo   ================================================
echo.
pause
