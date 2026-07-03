@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
echo Starting Stage 1 training... > run_p2_stage1.log
D:\Anaconda\envs\qwen3-code-lab\python.exe scripts\train_lora.py --config configs\curriculum\p2-stage1-code-continual.yaml >> run_p2_stage1.log 2>&1
echo. >> run_p2_stage1.log
echo Stage 1 exit code: %ERRORLEVEL% >> run_p2_stage1.log
