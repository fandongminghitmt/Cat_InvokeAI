@echo off
setlocal

:: ================= 配置区域 =================
:: 1. Python 环境路径 (借用已安装的 InvokeAI 环境)
set "PYTHON_ENV=d:\InvokeAI\.venv"

:: 2. 源代码路径 (您下载的新代码)
set "SOURCE_CODE=d:\CatInvoke\InvokeAI"

:: 3. 数据根目录 (存放模型和配置的地方)
set "DATA_ROOT=d:\InvokeAI"
:: ===========================================

:: 检查 Python 环境
if not exist "%PYTHON_ENV%\Scripts\python.exe" (
    echo [错误] 找不到 Python 环境，请确认路径正确: %PYTHON_ENV%
    pause
    exit /b
)

echo [状态] 正在启动 InvokeAI 源码模式...
echo [环境] %PYTHON_ENV%
echo [源码] %SOURCE_CODE%

:: 关键步骤：设置 PYTHONPATH，强制 Python 使用您的源代码目录
set "PYTHONPATH=%SOURCE_CODE%"

:: 启动命令
"%PYTHON_ENV%\Scripts\python.exe" "%SOURCE_CODE%\invokeai\app\run_app.py" --root "%DATA_ROOT%"

if %errorlevel% neq 0 (
    echo [警告] 程序异常退出，请检查上方错误信息。
)
pause