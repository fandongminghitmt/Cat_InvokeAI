@echo off

cd /d "D:\Cat_InvokeAI"

set "PYTHONPATH=D:\Cat_InvokeAI\invokeai"
set "INVOKEAI_ROOT=D:\Cat_InvokeAI"

"D:\Cat_InvokeAI\.venv\Scripts\python.exe" "D:\Cat_InvokeAI\.update_system\check_update.py"

echo [Starting] InvokeAI Source Mode...
echo Python Env: D:\Cat_InvokeAI\.venv
echo Source: D:\Cat_InvokeAI\invokeai

"D:\Cat_InvokeAI\.venv\Scripts\python.exe" "D:\Cat_InvokeAI\invokeai\invokeai\app\run_app.py"

pause