@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe
set DATASET=data\p2-curriculum\frozen-eval-v2\test_raw.jsonl
set MAX_SAMPLES=120

echo [Stage3] Started at %DATE% %TIME%
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --adapter adapters\p2\continual\stage3-repair-v2 --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\stage3-repair.json
echo [Stage3] exit code: %ERRORLEVEL%
echo [Stage3] Finished at %DATE% %TIME%
