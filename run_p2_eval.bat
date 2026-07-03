@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe
set DATASET=data\p2-curriculum\frozen-eval-v2\test_raw.jsonl
set MAX_SAMPLES=120

echo ============================================================
echo P2 Evaluation: 4 models on frozen-eval-v2 (max %MAX_SAMPLES% samples each)
echo ============================================================
echo Started at %DATE% %TIME% > run_p2_eval.log

echo. >> run_p2_eval.log
echo [1/4] Base Qwen3-0.6B >> run_p2_eval.log
echo --- Base --- >> run_p2_eval.log
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\base.json >> run_p2_eval.log 2>&1
echo Base exit code: %ERRORLEVEL% >> run_p2_eval.log

echo. >> run_p2_eval.log
echo [2/4] Stage 1 Code Adapter >> run_p2_eval.log
echo --- Stage1 --- >> run_p2_eval.log
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --adapter adapters\p2\continual\stage1-code-v2 --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\stage1-code.json >> run_p2_eval.log 2>&1
echo Stage1 exit code: %ERRORLEVEL% >> run_p2_eval.log

echo. >> run_p2_eval.log
echo [3/4] Stage 2 Boundary Adapter >> run_p2_eval.log
echo --- Stage2 --- >> run_p2_eval.log
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --adapter adapters\p2\continual\stage2-boundary-v2 --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\stage2-boundary.json >> run_p2_eval.log 2>&1
echo Stage2 exit code: %ERRORLEVEL% >> run_p2_eval.log

echo. >> run_p2_eval.log
echo [4/4] Stage 3 Repair Adapter >> run_p2_eval.log
echo --- Stage3 --- >> run_p2_eval.log
%PYTHON% scripts\evaluate_model.py --model models\Qwen3-0.6B --adapter adapters\p2\continual\stage3-repair-v2 --dataset %DATASET% --max-samples %MAX_SAMPLES% --output evaluations\p2\stage3-repair.json >> run_p2_eval.log 2>&1
echo Stage3 exit code: %ERRORLEVEL% >> run_p2_eval.log

echo. >> run_p2_eval.log
echo ============================================================
echo P2 Evaluation Complete at %DATE% %TIME% >> run_p2_eval.log
echo ============================================================
