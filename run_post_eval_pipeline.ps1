# run_post_eval_pipeline.ps1
# Waits for all 5 Full-576 eval files, then runs the full analysis pipeline.
cd e:\agent\Qwen\qwen3-code-lab

$py = "D:\Anaconda\envs\qwen3-code-lab\python.exe"
$eval_dir = "evaluations/p2"
$required = @(
    "full576-base.json",
    "full576-stage2-boundary.json",
    "full576-stage3-repair.json",
    "full576-independent-stage3.json",
    "full576-stage3-v3-antiforget.json"
)

Write-Host "=== Post-Eval Pipeline: Waiting for all 5 eval files ==="
foreach ($f in $required) {
    $path = Join-Path $eval_dir $f
    while (-not (Test-Path $path)) {
        Write-Host "  Waiting for $f ..."
        Start-Sleep -Seconds 60
    }
    Write-Host "  Found: $f"
}

Write-Host "=== All 5 eval files present. Starting pipeline. ==="
Write-Host "=== Start time: $(Get-Date) ==="

# Step 1: Compare evals
Write-Host "--- Step 1/4: compare_p2_evals.py ---"
& $py scripts/compare_p2_evals.py
Write-Host "compare exit: $LASTEXITCODE"

# Step 2: Paired stats
Write-Host "--- Step 2/4: compute_paired_stats.py ---"
& $py scripts/compute_paired_stats.py
Write-Host "paired_stats exit: $LASTEXITCODE"

# Step 3: Router analysis (includes P3 Decision Gate)
Write-Host "--- Step 3/4: compute_router_analysis.py ---"
& $py scripts/compute_router_analysis.py
Write-Host "router_analysis exit: $LASTEXITCODE"

# Step 4: Full-576 comparison report
Write-Host "--- Step 4/4: generate_full576_report.py ---"
& $py scripts/generate_full576_report.py
Write-Host "report exit: $LASTEXITCODE"

Write-Host "=== Pipeline complete: $(Get-Date) ==="
Write-Host "=== Generated files: ==="
Write-Host "  evaluations/p2/full576-comparison.json"
Write-Host "  reports/p2/full576-paired-stats.json"
Write-Host "  reports/p2/full576-paired-stats.md"
Write-Host "  reports/p2/router-analysis.json"
Write-Host "  reports/p2/router-analysis.md"
Write-Host "  reports/p2/p2-full576-comparison-report.md"
Write-Host "POST_EVAL_PIPELINE_DONE_MARKER"
