@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe
set DATASET=data\p2-curriculum\frozen-eval-v2\test_raw.jsonl
set MAX_SAMPLES=120

echo [Stage1] Started at %DATE% %TIME%
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --adapter adapters\p2\continual\stage1-code-v2 --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\stage1-code.json
echo [Stage1] exit code: %ERRORLEVEL%
echo [Stage1] Finished at %DATE% %TIME%
