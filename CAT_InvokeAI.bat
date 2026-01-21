@echo off

set "PYTHONPATH=%~dp0invokeai"
set "INVOKEAI_ROOT=%~dp0"

REM Fix venv path dynamically
"%~dp0uv\python\cpython-3.12.9-windows-x86_64-none\python.exe" "%~dp0fix_path.py"

"%~dp0.venv\Scripts\python.exe" "%~dp0.update_system\check_update.py"

echo [Starting] InvokeAI Source Mode...
echo Python Env: .venv
echo Source: invokeai

"%~dp0.venv\Scripts\python.exe" "%~dp0invokeai\invokeai\app\run_app.py"

pause
