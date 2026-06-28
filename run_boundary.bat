@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
D:\Anaconda\envs\qwen3-code-lab\python.exe scripts\train_lora.py --config configs\train_boundary.yaml > train_boundary.log 2>&1
echo Exit code: %ERRORLEVEL% >> train_boundary.log