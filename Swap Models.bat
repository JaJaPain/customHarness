@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" model_swap.py %*
) else (
  start "" pythonw model_swap.py %*
)
