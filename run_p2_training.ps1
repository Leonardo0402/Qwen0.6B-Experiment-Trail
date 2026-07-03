# P2 Three-stage Continual Training Launcher
# Bypasses TRAE sandbox for long-running Python processes
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$ROOT = "e:\agent\Qwen\qwen3-code-lab"
$PYTHON = "D:\Anaconda\envs\qwen3-code-lab\python.exe"
$LOG = Join-Path $ROOT "run_p2_training_full.log"

Set-Location $ROOT

function Run-Stage {
    param([string]$name, [string]$config)
    $header = "`n============================================================`n$name`n============================================================"
    Write-Output $header
    Add-Content -Path $LOG -Value $header
    $start = Get-Date
    & $PYTHON "scripts\train_lora.py" "--config" $config 2>&1 | Tee-Object -FilePath $LOG -Append
    $exitCode = $LASTEXITCODE
    $elapsed = ((Get-Date) - $start).TotalSeconds
    $footer = "Stage finished in $([math]::Round($elapsed, 1))s with exit code $exitCode"
    Write-Output $footer
    Add-Content -Path $LOG -Value $footer
    if ($exitCode -ne 0) {
        Write-Output "STAGE FAILED: $name (exit $exitCode)"
        exit $exitCode
    }
}

# Clear previous log
Set-Content -Path $LOG -Value "P2 Training Log - $(Get-Date)"

Run-Stage "P2 Stage 1: Code Foundation (Independent from base)" "configs\curriculum\p2-stage1-code-continual.yaml"
Run-Stage "P2 Stage 2: Boundary Reasoning (Continual from Stage 1)" "configs\curriculum\p2-stage2-boundary-continual.yaml"
Run-Stage "P2 Stage 3: Execution Repair (Continual from Stage 2)" "configs\curriculum\p2-stage3-repair-continual.yaml"

$done = "`n============================================================`nP2 Training Complete`n============================================================"
Write-Output $done
Add-Content -Path $LOG -Value $done
