@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
D:\Anaconda\envs\qwen3-code-lab\python.exe -X faulthandler -u scripts\evaluate_model.py --model models/Qwen3-0.6B --adapter adapters/p2/independent/stage3-repair-v2 --dataset data/p2-curriculum/frozen-eval-v2/stratified-120/test_raw.jsonl --output evaluations/p2/independent-stage3.json > eval_p1_debug.log 2>&1
echo P1_EVAL_EXIT_CODE=%ERRORLEVEL% >> eval_p1_debug.log
echo P1_EVAL_DONE_MARKER >> eval_p1_debug.log
