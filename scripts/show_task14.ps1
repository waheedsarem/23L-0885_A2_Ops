param([ValidateSet('toy','dataset','recovery')][string]$Section = 'toy')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dvc = Join-Path $projectRoot '.venv\Scripts\dvc.exe'
$toy = Join-Path $projectRoot 'evidence\task_1_4\toy_repo'
$windowLabel = if ($Section -eq 'recovery') { 'recovery-final' } else { $Section }
$Host.UI.RawUI.WindowTitle = "23L-0885_A2_Ops - Task 1.4 - $windowLabel"
try {
    $buffer = $Host.UI.RawUI.BufferSize
    $buffer.Width = 145
    $buffer.Height = 1000
    $Host.UI.RawUI.BufferSize = $buffer
    $window = $Host.UI.RawUI.WindowSize
    $window.Width = 145
    $window.Height = 40
    $Host.UI.RawUI.WindowSize = $window
} catch { }
Clear-Host
Write-Host 'VisionGuard | 23L-0885_A2_Ops | Task 1.4' -ForegroundColor Cyan
Write-Host "User: $env:USERNAME | Project: $projectRoot"
Write-Host "DagsHub: https://dagshub.com/waheedsarem/23L-0885_A2_Ops"
Write-Host "Recorded: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')) (Asia/Karachi)"
Write-Host

if ($Section -eq 'toy') {
    Set-Location -LiteralPath $toy
    Write-Host 'TOY CSV: TWO GIT/DVC VERSIONS' -ForegroundColor Yellow
    git log --oneline --decorate -2
    Write-Host
    Write-Host 'Version 1 pointer (commit b3c76ce):'
    git show 'b3c76ce:data/toy.csv.dvc'
    Write-Host 'Version 2 pointer (commit 26335d6):'
    git show '26335d6:data/toy.csv.dvc'
    Write-Host 'Recovered current CSV:'
    Get-Content data\toy.csv
    Write-Host "Recovered MD5: $((Get-FileHash data\toy.csv -Algorithm MD5).Hash.ToLower())"
    Write-Host "Separate remote object files: $((Get-ChildItem ..\toy_remote -File -Recurse).Count)"
    Write-Host 'Remote Git branch:'
    git ls-remote origin refs/heads/dvc-toy
} elseif ($Section -eq 'dataset') {
    Set-Location -LiteralPath $projectRoot
    $env:DVC_SITE_CACHE_DIR = Join-Path $projectRoot '.dvc\site-cache'
    Write-Host 'ACTUAL DATASET: DVC POINTER AND DAGSHUB SYNC' -ForegroundColor Yellow
    git log --oneline --decorate -3
    Write-Host
    Get-Content dataset\raw.dvc
    Write-Host 'Configured non-secret DVC remote:'
    & $dvc remote list
    Write-Host 'DagsHub storage comparison:'
    & $dvc status -c
    $files = Get-ChildItem dataset\raw -File -Recurse
    Write-Host "Recovered/tracked dataset: $($files.Count) files, $(($files | Measure-Object Length -Sum).Sum) bytes"
    Write-Host "config.local ignored: $([bool](git check-ignore .dvc\config.local))"
    Write-Host 'Remote Git branches:'
    git ls-remote origin refs/heads/main refs/heads/dvc-toy
} else {
    Set-Location -LiteralPath $toy
    $env:DVC_SITE_CACHE_DIR = Join-Path $toy '.dvc\site-cache'
    $data = Join-Path $toy 'data\toy.csv'
    $cache = Join-Path $toy '.dvc\cache\files\md5\a6\3bbadcd07bbedd9b44379e7243c4bb'
    $remote = Join-Path (Split-Path $toy -Parent) 'toy_remote\files\md5\a6\3bbadcd07bbedd9b44379e7243c4bb'
    Write-Host 'REMOTE-ONLY RECOVERY PROOF' -ForegroundColor Yellow
    if (-not (Test-Path -LiteralPath $remote)) { throw 'Remote object is missing.' }
    & $dvc unprotect data\toy.csv
    if (Test-Path -LiteralPath $data) { Remove-Item -LiteralPath $data }
    if (Test-Path -LiteralPath $cache) {
        Set-ItemProperty -LiteralPath $cache -Name IsReadOnly -Value $false
        Remove-Item -LiteralPath $cache
    }
    $remoteRetained = Test-Path -LiteralPath $remote
    $dataAfterDelete = Test-Path -LiteralPath $data
    $cacheAfterDelete = Test-Path -LiteralPath $cache
    & $dvc -q pull -r localstore | Out-Null
    $dataAfterPull = Test-Path -LiteralPath $data
    $cacheAfterPull = Test-Path -LiteralPath $cache
    $hash = (Get-FileHash data\toy.csv -Algorithm MD5).Hash.ToLower()
    Clear-Host
    Write-Host 'VisionGuard | 23L-0885_A2_Ops | Task 1.4' -ForegroundColor Cyan
    Write-Host "User: $env:USERNAME | Project: $projectRoot"
    Write-Host "Recorded: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')) (Asia/Karachi)"
    Write-Host
    Write-Host 'REMOTE-ONLY RECOVERY PROOF' -ForegroundColor Yellow
    Write-Host "Remote object retained: $remoteRetained"
    Write-Host "Local CSV after delete: $dataAfterDelete"
    Write-Host "Local cache object after delete: $cacheAfterDelete"
    Write-Host "Local CSV after remote pull: $dataAfterPull"
    Write-Host "Local cache after remote pull: $cacheAfterPull"
    Write-Host
    Write-Host 'Recovered CSV:'
    Get-Content data\toy.csv
    Write-Host "Recovered MD5: $hash"
    if ($hash -ne 'a63bbadcd07bbedd9b44379e7243c4bb') { throw 'Recovered hash mismatch.' }
    Write-Host 'REMOTE RECOVERY: PASS' -ForegroundColor Green
}
Write-Host "`nExit code: $LASTEXITCODE"
