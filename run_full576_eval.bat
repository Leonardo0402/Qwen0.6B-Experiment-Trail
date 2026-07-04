@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PY=D:\Anaconda\envs\qwen3-code-lab\python.exe
set DATASET=data/p2-curriculum/frozen-eval-v2/test_raw.jsonl
set MODEL=models/Qwen3-0.6B

echo === Full-576 Evaluation Start: %DATE% %TIME% ===

echo --- Model 1/5: Base ---
%PY% -X faulthandler -u scripts\evaluate_model.py --model %MODEL% --dataset %DATASET% --output evaluations/p2/full576-base.json 2>&1
echo Base exit: %ERRORLEVEL%

echo --- Model 2/5: Stage2-v2 ---
%PY% -X faulthandler -u scripts\evaluate_model.py --model %MODEL% --adapter adapters/p2/continual/stage2-boundary-v2 --dataset %DATASET% --output evaluations/p2/full576-stage2-boundary.json 2>&1
echo Stage2 exit: %ERRORLEVEL%

echo --- Model 3/5: Stage3-v2-Continual ---
%PY% -X faulthandler -u scripts\evaluate_model.py --model %MODEL% --adapter adapters/p2/continual/stage3-repair-v2 --dataset %DATASET% --output evaluations/p2/full576-stage3-repair.json 2>&1
echo Stage3-v2 exit: %ERRORLEVEL%

echo --- Model 4/5: Stage3-Independent ---
%PY% -X faulthandler -u scripts\evaluate_model.py --model %MODEL% --adapter adapters/p2/independent/stage3-repair-v2 --dataset %DATASET% --output evaluations/p2/full576-independent-stage3.json 2>&1
echo Independent exit: %ERRORLEVEL%

echo --- Model 5/5: Stage3-v3-Antiforget ---
%PY% -X faulthandler -u scripts\evaluate_model.py --model %MODEL% --adapter adapters/p2/continual/stage3-repair-v3 --dataset %DATASET% --output evaluations/p2/full576-stage3-v3-antiforget.json 2>&1
echo Antiforget exit: %ERRORLEVEL%

echo === Full-576 Evaluation Complete: %DATE% %TIME% ===
echo FULL576_ALL_DONE_MARKER
