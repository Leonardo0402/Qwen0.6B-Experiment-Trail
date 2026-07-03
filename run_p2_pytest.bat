@echo off
cd /d e:\agent\Qwen\qwen3-code-lab
set PYTHONUNBUFFERED=1
set PYTHON=D:\Anaconda\envs\qwen3-code-lab\python.exe

echo P2 Pytest Suite started at %DATE% %TIME% > run_p2_pytest.log
echo ============================================================ >> run_p2_pytest.log
%PYTHON% -m pytest tests/ --tb=short -q >> run_p2_pytest.log 2>&1
echo Pytest exit code: %ERRORLEVEL% >> run_p2_pytest.log
echo ============================================================ >> run_p2_pytest.log
echo P2 Pytest Suite finished at %DATE% %TIME% >> run_p2_pytest.log
