@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe

echo ============================================================
echo P2 v2 Training: 3-stage Continual LoRA (with function signatures)
echo ============================================================
echo Started at %DATE% %TIME% > run_p2_training_v2.log

echo. >> run_p2_training_v2.log
echo [1/3] Stage 1: Code Foundation (Independent) >> run_p2_training_v2.log
echo P2 v2 Stage 1 starting at %DATE% %TIME% > run_p2_stage1_v2.log
%PYTHON% scripts\train_lora.py --config configs\curriculum\p2-stage1-code-continual.yaml >> run_p2_stage1_v2.log 2>&1
echo Stage 1 exit code: %ERRORLEVEL% >> run_p2_stage1_v2.log
if %ERRORLEVEL% neq 0 (
    echo STAGE 1 FAILED with code %ERRORLEVEL%
    exit /b 1
)

echo. >> run_p2_training_v2.log
echo [2/3] Stage 2: Boundary Reasoning (Continual from Stage 1) >> run_p2_training_v2.log
echo P2 v2 Stage 2 starting at %DATE% %TIME% > run_p2_stage2_v2.log
%PYTHON% scripts\train_lora.py --config configs\curriculum\p2-stage2-boundary-continual.yaml >> run_p2_stage2_v2.log 2>&1
echo Stage 2 exit code: %ERRORLEVEL% >> run_p2_stage2_v2.log
if %ERRORLEVEL% neq 0 (
    echo STAGE 2 FAILED with code %ERRORLEVEL%
    exit /b 2
)

echo. >> run_p2_training_v2.log
echo [3/3] Stage 3: Execution Repair (Continual from Stage 2) >> run_p2_training_v2.log
echo P2 v2 Stage 3 starting at %DATE% %TIME% > run_p2_stage3_v2.log
%PYTHON% scripts\train_lora.py --config configs\curriculum\p2-stage3-repair-continual.yaml >> run_p2_stage3_v2.log 2>&1
echo Stage 3 exit code: %ERRORLEVEL% >> run_p2_stage3_v2.log
if %ERRORLEVEL% neq 0 (
    echo STAGE 3 FAILED with code %ERRORLEVEL%
    exit /b 3
)

echo. >> run_p2_training_v2.log
echo ============================================================
echo P2 v2 All Stages Complete at %DATE% %TIME% >> run_p2_training_v2.log
echo ============================================================
