@echo off
echo ==========================================
echo      BERT Sentiment Classification Trainer
echo ==========================================

set HF_ENDPOINT=https://hf-mirror.com
echo [Info] Set HF_ENDPOINT=https://hf-mirror.com

echo [Info] Trying to activate Conda environment: AD ...
call conda activate AD 2>nul
if %errorlevel% equ 0 (
    echo [Success] Activated environment: AD
) else (
    echo [Warning] Automatic activation failed, using current environment.
)

echo.
echo [Info] Current Python:
where python
echo.

echo [Info] Starting Trainer script...
cd D:\demo\data-mining-and-knowledge-processing-main\2025-spring\exp02-sentiment-classificationn\bert-sentential-classifer\
python main_trainer.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] Program execution failed!
    pause
    exit /b 1
)

echo.
echo [Success] Program execution completed.
pause
