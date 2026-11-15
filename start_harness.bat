@echo off
title Godot 4 Local Agentic Coding and 2D/3D Asset Harness
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

if exist "D:\Program Files\Blender Foundation\Blender 5.2\blender.exe" set "BLENDER_PATH=D:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
if exist "D:\CodingProjects\spacegame\Godot\Godot_v4.6.3-stable_win64_console.exe" set "GODOT_PATH=D:\CodingProjects\spacegame\Godot\Godot_v4.6.3-stable_win64_console.exe"
if exist "C:\HarnessModels" set "HARNESS_FAST_MODELS_DIR=C:\HarnessModels"
set "PATH=%~dp0bin;D:\Program Files\Blender Foundation\Blender 5.2;D:\CodingProjects\spacegame\Godot;%PATH%"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

%PYTHON_EXE% --version >nul 2>nul
if errorlevel 1 goto :no_python

echo [1/2] Checking core Python dependencies...
%PYTHON_EXE% -c "import fastapi, uvicorn, websockets, httpx, pydantic, psutil, PIL, numpy" >nul 2>nul
if errorlevel 1 goto :install_deps
goto :launch_server

:install_deps
echo [1/2] Installing required Python packages from requirements.txt...
%PYTHON_EXE% -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed

:launch_server
echo [2/2] Starting FastAPI + WebSockets Harness Server on http://127.0.0.1:8000 ...
echo       (Your browser will open automatically once the server is listening)
echo.
set "HARNESS_OPEN_BROWSER=1"
%PYTHON_EXE% -u server.py
echo.
echo [INFO] Server stopped.
pause
exit /b 0

:no_python
echo [ERROR] Python 3.10+ was not found on your PATH.
echo Please install Python 3.10+ for Windows and ensure it is added to PATH.
pause
exit /b 1

:pip_failed
echo [ERROR] Failed to install required Python dependencies.
pause
exit /b 1
