@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
D:\Anaconda\envs\qwen3-code-lab\python.exe scripts\train_lora.py --config configs\curriculum\p2-stage3-repair-independent.yaml > p1_train_debug.log 2>&1
echo P1_TRAIN_EXIT_CODE=%ERRORLEVEL% >> p1_train_debug.log
echo P1_TRAIN_DONE_MARKER >> p1_train_debug.log
