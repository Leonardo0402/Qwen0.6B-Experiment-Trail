@echo off
cd /d "e:\agent\Qwen\qwen3-code-lab"
git push -u origin main --verbose 2>&1
echo.
echo === Push completed with exit code: %ERRORLEVEL% ===
pause