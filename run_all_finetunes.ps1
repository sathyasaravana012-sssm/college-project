$ErrorActionPreference = "Continue"
$py = "venv\Scripts\python.exe"

# Only models in config.BASELINE_MODELS are accepted by finetune_plantdoc.py.
# Plain resnet18 was retired from that set (see the comment there), so it is no
# longer a job here. Its checkpoint and evaluate_resnet18.py remain in the repo
# if the ablation needs re-running.
$jobs = @(
    @{ model = "basic";             pv = 150; epochs = 5 },
    @{ model = "mobilenetv2";       pv = 150; epochs = 5 },
    @{ model = "resnet18_improved"; pv = 100; epochs = 4 }
)

foreach ($job in $jobs) {
    $m = $job.model
    Write-Output ""
    Write-Output "############################################################"
    Write-Output "# START $m at $(Get-Date -Format 'HH:mm:ss')"
    Write-Output "############################################################"

    & $py finetune_plantdoc.py --model $m --epochs $job.epochs `
        --plantvillage-per-class $job.pv --plantdoc-weight 2 --val-per-class 30 `
        2>&1 | Tee-Object -FilePath "logs_ft_$m.txt" | Out-Null

    Write-Output "# END $m at $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE"
}

Write-Output ""
Write-Output "ALL FINETUNE JOBS COMPLETE at $(Get-Date -Format 'HH:mm:ss')"