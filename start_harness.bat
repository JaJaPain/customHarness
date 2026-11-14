@echo off
setlocal enabledelayedexpansion
title Godot 4 Local Agentic Coding ^& 2D/3D Asset Harness (RTX 3060 12GB)

cd /d "%~dp0"

echo ============================================================================
echo   GODOT 4 STANDALONE LOCAL-FIRST AGENTIC CODING ^& 2D/3D ASSET HARNESS
echo   Target Hardware: Windows 11 ^| NVIDIA RTX 3060 12GB VRAM ^| 32GB DDR4 RAM
echo   Engines: llama-server.exe (CUDA) + sd.exe (FLUX.1-schnell) + Blender 4.x
echo ============================================================================
echo.

if not exist "bin" mkdir "bin"
if not exist "models" mkdir "models"
if not exist "cache\slots" mkdir "cache\slots"
if not exist "cache\kv_disk" mkdir "cache\kv_disk"
if not exist "workspace" mkdir "workspace"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10+ not found in PATH. Please install Python 3.10+ for Windows.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating local Python virtual environment (.venv)...
    python -m venv .venv
)

echo [2/3] Verifying Python dependencies from requirements.txt...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

echo [3/3] Starting FastAPI + WebSockets Harness Server on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
python server.py

pause
