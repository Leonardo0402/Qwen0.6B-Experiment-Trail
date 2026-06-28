@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe
set DATASET=data\splits\test_raw.jsonl
set OUT=evaluations\fixed-p0
set MODEL=models\Qwen3-0.6B

if not exist %OUT% mkdir %OUT%

echo ============================================================
echo [1/4] Evaluating Baseline
echo ============================================================
%PYTHON% scripts\evaluate_model.py --model %MODEL% --dataset %DATASET% --output %OUT%\baseline.json
echo Baseline exit code: %ERRORLEVEL%
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: Baseline evaluation
    exit /b 1
)

echo ============================================================
echo [2/4] Evaluating v3-easy
echo ============================================================
%PYTHON% scripts\evaluate_model.py --model %MODEL% --adapter adapters\code-lora-v3-easy --dataset %DATASET% --output %OUT%\v3-easy.json
echo v3-easy exit code: %ERRORLEVEL%
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: v3-easy evaluation
    exit /b 2
)

echo ============================================================
echo [3/4] Evaluating v3-boundary-v2
echo ============================================================
%PYTHON% scripts\evaluate_model.py --model %MODEL% --adapter adapters\code-lora-v3-boundary-v2 --dataset %DATASET% --output %OUT%\v3-boundary-v2.json
echo v3-boundary-v2 exit code: %ERRORLEVEL%
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: v3-boundary-v2 evaluation
    exit /b 3
)

echo ============================================================
echo [4/4] Evaluating v3-repair
echo ============================================================
%PYTHON% scripts\evaluate_model.py --model %MODEL% --adapter adapters\code-lora-v3-repair --dataset %DATASET% --output %OUT%\v3-repair.json
echo v3-repair exit code: %ERRORLEVEL%
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: v3-repair evaluation
    exit /b 4
)

echo ============================================================
echo ALL EVALUATIONS COMPLETE
echo ============================================================
exit /b 0
