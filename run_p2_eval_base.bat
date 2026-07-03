@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe
set DATASET=data\p2-curriculum\frozen-eval-v2\test_raw.jsonl
set MAX_SAMPLES=120

echo [Base] Started at %DATE% %TIME%
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\base.json
echo [Base] exit code: %ERRORLEVEL%
echo [Base] Finished at %DATE% %TIME%
