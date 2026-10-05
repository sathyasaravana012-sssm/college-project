$ErrorActionPreference = "Continue"
$py = "venv\Scripts\python.exe"

$jobs = @(
    @{ model = "basic";          pv = 150; epochs = 5 },
    @{ model = "resnet18_improved"; pv = 100; epochs = 4 },
    @{ model = "resnet18";       pv = 100; epochs = 4 }
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