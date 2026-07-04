# run_remaining_evals.ps1
# Waits for Base Full-576 eval to finish, then runs the remaining 4 models.
cd e:\agent\Qwen\qwen3-code-lab

$base_output = "evaluations/p2/full576-base.json"
$py = "D:\Anaconda\envs\qwen3-code-lab\python.exe"
$dataset = "data/p2-curriculum/frozen-eval-v2/test_raw.jsonl"
$model = "models/Qwen3-0.6B"

Write-Host "=== Waiting for Base eval to complete (checking for $base_output) ==="
while (-not (Test-Path $base_output)) {
    Start-Sleep -Seconds 30
}
Write-Host "=== Base eval complete. Starting remaining 4 models ==="
Write-Host "=== Start time: $(Get-Date) ==="

# Model 2/5: Stage2-v2
Write-Host "--- Model 2/5: Stage2-v2 ---"
& $py -X faulthandler -u scripts/evaluate_model.py --model $model --adapter adapters/p2/continual/stage2-boundary-v2 --dataset $dataset --output evaluations/p2/full576-stage2-boundary.json
Write-Host "Stage2 exit: $LASTEXITCODE"

# Model 3/5: Stage3-v2-Continual
Write-Host "--- Model 3/5: Stage3-v2-Continual ---"
& $py -X faulthandler -u scripts/evaluate_model.py --model $model --adapter adapters/p2/continual/stage3-repair-v2 --dataset $dataset --output evaluations/p2/full576-stage3-repair.json
Write-Host "Stage3-v2 exit: $LASTEXITCODE"

# Model 4/5: Stage3-Independent
Write-Host "--- Model 4/5: Stage3-Independent ---"
& $py -X faulthandler -u scripts/evaluate_model.py --model $model --adapter adapters/p2/independent/stage3-repair-v2 --dataset $dataset --output evaluations/p2/full576-independent-stage3.json
Write-Host "Independent exit: $LASTEXITCODE"

# Model 5/5: Stage3-v3-Antiforget
Write-Host "--- Model 5/5: Stage3-v3-Antiforget ---"
& $py -X faulthandler -u scripts/evaluate_model.py --model $model --adapter adapters/p2/continual/stage3-repair-v3 --dataset $dataset --output evaluations/p2/full576-stage3-v3-antiforget.json
Write-Host "Antiforget exit: $LASTEXITCODE"

Write-Host "=== All remaining evals complete: $(Get-Date) ==="
Write-Host "ALL_REMAINING_EVALS_DONE_MARKER"
