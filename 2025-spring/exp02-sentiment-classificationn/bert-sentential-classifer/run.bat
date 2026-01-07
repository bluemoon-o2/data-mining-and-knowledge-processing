@echo off
chcp 65001 > nul
echo ==========================================
echo      BERT Sentiment Classification
echo ==========================================

:: 设置 HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com
echo [Info] 已设置 HF_ENDPOINT=https://hf-mirror.com

:: 尝试激活 Conda 环境
echo [Info] 尝试激活 Conda 环境: AD ...
call conda activate AD 2>nul
if %errorlevel% equ 0 (
    echo [Success] 已激活环境: AD
) else (
    echo [Warning] 自动激活失败，将使用当前环境。
)

echo.
echo [Info] 当前使用的 Python:
where python
echo.

echo [Info] 开始运行训练脚本...
cd /d %~dp0
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] 程序运行出错！
    pause
    exit /b 1
)

echo.
echo [Success] 程序运行完成。
pause
