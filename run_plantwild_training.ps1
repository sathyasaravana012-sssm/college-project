# Runs the full PlantWild training for all four models, sequentially.
# Logs go to C:\Plant_disease_detection\logs\pw_<step>_<stamp>.txt
# Marker: logs\pw_RUN_COMPLETE.txt is written when everything finishes.

$root = 'C:\Plant_disease_detection'
$py   = Join-Path $root 'venv\Scripts\python.exe'
$dino = Join-Path $root 'dinov2_experiment\.venv\Scripts\python.exe'
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$results = @()

function Invoke-Logged {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$ArgList,
        [string]$WorkDir = $root,
        [hashtable]$EnvVars = @{}
    )
    $log = Join-Path $logs ("pw_{0}_{1}.txt" -f $Name, $stamp)
    "=== $Name === start $(Get-Date -Format s)" | Set-Content -LiteralPath $log
    # clear then set experiment-specific env vars
    Remove-Item -Path env:PDD_DATASET -ErrorAction SilentlyContinue
    foreach ($k in $EnvVars.Keys) {
        Set-Item -Path ("env:{0}" -f $k) -Value $EnvVars[$k]
    }
    Push-Location -LiteralPath $WorkDir
    try {
        & $Exe @ArgList 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        Remove-Item -Path env:PDD_DATASET -ErrorAction SilentlyContinue
    }
    "=== exit code: $code === end $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append | Out-Null
    $script:results += [pscustomobject]@{ Step = $Name; ExitCode = $code; Log = $log }
    return $code
}

# ---------------- 1-3: TensorFlow models ----------------

Invoke-Logged -Name 'simple_cnn' -Exe $py `
    -ArgList @((Join-Path $root 'train_plantwild_simple_cnn.py'))

Invoke-Logged -Name 'resnet18' -Exe $py `
    -ArgList @((Join-Path $root 'train_plantwild_resnet18.py'))

Invoke-Logged -Name 'mobilenetv2' -Exe $py `
    -ArgList @((Join-Path $root 'train_plantwild_mobilenetv2.py'))

# ---------------- 4: DINOv2 pipeline ----------------

Invoke-Logged -Name 'dinov2_extract' -Exe $dino `
    -ArgList @('extract_features.py', '--splits', 'train', 'validation', 'test',
               '--batch-size', '32', '--overwrite') `
    -WorkDir (Join-Path $root 'dinov2_experiment') `
    -EnvVars @{ PDD_DATASET = 'plantwild' }

Invoke-Logged -Name 'dinov2_train' -Exe $dino `
    -ArgList @('train_classifier.py') `
    -WorkDir (Join-Path $root 'dinov2_experiment') `
    -EnvVars @{ PDD_DATASET = 'plantwild' }

Invoke-Logged -Name 'dinov2_test_eval' -Exe $dino `
    -ArgList @('eval_test.py') `
    -WorkDir (Join-Path $root 'dinov2_experiment') `
    -EnvVars @{ PDD_DATASET = 'plantwild' }

# ---------------- done ----------------

$summary = Join-Path $logs 'pw_RUN_COMPLETE.txt'
"PlantWild training run finished at $(Get-Date -Format s)" | Set-Content -LiteralPath $summary
$results | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $summary -Append | Out-Null
