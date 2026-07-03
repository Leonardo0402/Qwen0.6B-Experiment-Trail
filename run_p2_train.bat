@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
D:\Anaconda\envs\qwen3-code-lab\python.exe scripts\train_lora.py --config configs\curriculum\p2-stage3-repair-v3-antiforget.yaml > p2_train_debug.log 2>&1
echo P2_TRAIN_EXIT_CODE=%ERRORLEVEL% >> p2_train_debug.log
echo P2_TRAIN_DONE_MARKER >> p2_train_debug.log
