# Issue #1 Remainder Runner: P1 + P2 train/eval + stats + reports
# Waits for v3 stratified-120 eval to finish, then sequentially:
#   1. Train P1 Independent Stage3
#   2. Eval P1 on stratified-120
#   3. Train P2 Anti-forget Stage3-v3
#   4. Eval P2 on stratified-120
#   5. Re-compute paired stats with all 5 models
#   6. Regenerate comparison.json + adapter evidence
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$ROOT = "e:\agent\Qwen\qwen3-code-lab"
$PYTHON = "D:\Anaconda\envs\qwen3-code-lab\python.exe"
$LOG = Join-Path $ROOT "run_p2_issue1_remainder.log"
$DATASET = "data\p2-curriculum\frozen-eval-v2\stratified-120\test_raw.jsonl"

Set-Location $ROOT
Set-Content -Path $LOG -Value "Issue #1 Remainder Runner - $(Get-Date)"

function Run-Step {
    param([string]$name, [string[]]$pyArgs, [int]$timeoutMin = 90)
    $header = "`n============================================================`n$name`nstarted=$(Get-Date)`n============================================================"
    Write-Output $header
    Add-Content -Path $LOG -Value $header
    $start = Get-Date
    $p = Start-Process -FilePath $PYTHON -ArgumentList ($pyArgs | ForEach-Object { "`"$_`"" }) `
        -RedirectStandardOutput "$ROOT\.step_stdout.tmp" `
        -RedirectStandardError "$ROOT\.step_stderr.tmp" `
        -NoNewWindow -PassThru
    while (-not $p.HasExited) {
        $elapsed = ((Get-Date) - $start).TotalMinutes
        if ($elapsed -gt $timeoutMin) {
            Write-Output "TIMEOUT after $timeoutMin min for $name - killing"
            try { $p.Kill() } catch {}
            Add-Content -Path $LOG -Value "TIMEOUT $name"
            return $false
        }
        Start-Sleep -Seconds 30
    }
    Get-Content "$ROOT\.step_stdout.tmp" -ErrorAction SilentlyContinue | Tee-Object -FilePath $LOG -Append | Out-Null
    Get-Content "$ROOT\.step_stderr.tmp" -ErrorAction SilentlyContinue | Tee-Object -FilePath $LOG -Append | Out-Null
    $exitCode = $p.ExitCode
    $elapsed = ((Get-Date) - $start).TotalSeconds
    $footer = "Step finished in $([math]::Round($elapsed, 1))s with exit code $exitCode"
    Write-Output $footer
    Add-Content -Path $LOG -Value $footer
    if ($exitCode -ne 0) {
        Write-Output "STEP FAILED: $name (exit $exitCode)"
        Add-Content -Path $LOG -Value "STEP FAILED: $name"
        return $false
    }
    return $true
}

# Step 0: Wait for v3 eval to finish (max 4 hours)
Write-Output "Waiting for v3 stratified-120 eval to complete..."
Add-Content -Path $LOG -Value "Waiting for v3 stratified-120 eval to complete..."
$maxMinutes = 240
$start = Get-Date
$found = $false
while (-not $found) {
    $elapsed = ((Get-Date) - $start).TotalMinutes
    if ($elapsed -gt $maxMinutes) {
        Write-Output "TIMEOUT after $maxMinutes minutes - v3 eval did not complete"
        Add-Content -Path $LOG -Value "TIMEOUT after $maxMinutes minutes"
        exit 1
    }
    $content = Get-Content "$ROOT\run_p2_eval_v3.log" -ErrorAction SilentlyContinue
    if ($content -match "P2 Re-eval Complete") {
        Write-Output "v3 eval complete at $([math]::Round($elapsed,1)) min"
        Add-Content -Path $LOG -Value "v3 eval complete at $([math]::Round($elapsed,1)) min"
        $found = $true
    } else {
        $tail = $content | Select-Object -Last 1
        Write-Output "[$([math]::Round($elapsed,0))min] waiting... tail: $tail"
        Start-Sleep -Seconds 60
    }
}

# Step 1: Train P1 Independent Stage3
$ok = Run-Step "P1: Train Independent Stage3 (from Base)" `
    @("scripts\train_lora.py", "--config", "configs\curriculum\p2-stage3-repair-independent.yaml") `
    -timeoutMin 90
if (-not $ok) { exit 1 }

# Step 2: Eval P1 on stratified-120
$ok = Run-Step "P1: Eval Independent Stage3 on stratified-120" `
    @("scripts\evaluate_model.py", "--model", "models\Qwen3-0.6B", `
        "--adapter", "adapters\p2\independent\stage3-repair-v2", `
        "--dataset", $DATASET, `
        "--output", "evaluations\p2\independent-stage3.json") `
    -timeoutMin 60
if (-not $ok) { exit 1 }

# Step 3: Train P2 Anti-forget Stage3-v3
$ok = Run-Step "P2: Train Anti-forget Stage3-v3 (from Stage2-v2)" `
    @("scripts\train_lora.py", "--config", "configs\curriculum\p2-stage3-repair-v3-antiforget.yaml") `
    -timeoutMin 90
if (-not $ok) { exit 1 }

# Step 4: Eval P2 on stratified-120
$ok = Run-Step "P2: Eval Stage3-v3 on stratified-120" `
    @("scripts\evaluate_model.py", "--model", "models\Qwen3-0.6B", `
        "--adapter", "adapters\p2\continual\stage3-repair-v3", `
        "--dataset", $DATASET, `
        "--output", "evaluations\p2\stage3-v3-antiforget.json") `
    -timeoutMin 60
if (-not $ok) { exit 1 }

# Step 5: Regenerate comparison.json
Run-Step "Regenerate comparison.json" `
    @("scripts\compare_p2_evals.py") `
    -timeoutMin 5 | Out-Null

# Step 6: Re-run paired stats with all 5 models
Run-Step "Re-run paired stats (P3, 5 models)" `
    @("scripts\compute_paired_stats.py") `
    -timeoutMin 5 | Out-Null

# Step 7: Recompute adapter evidence
Run-Step "Recompute adapter evidence" `
    @("scripts\compute_adapter_evidence.py") `
    -timeoutMin 5 | Out-Null

$done = "`n============================================================`nIssue #1 Remainder Complete at $(Get-Date)`n============================================================"
Write-Output $done
Add-Content -Path $LOG -Value $done
