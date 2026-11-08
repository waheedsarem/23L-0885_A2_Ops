param([ValidateSet('counts','statistics')][string]$Section = 'counts')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$Host.UI.RawUI.WindowTitle = "23L-0885_A2_Ops - Task 1.2 - $Section"
try {
    $size = $Host.UI.RawUI.BufferSize
    $size.Width = 140
    $size.Height = 1000
    $Host.UI.RawUI.BufferSize = $size
    $windowSize = $Host.UI.RawUI.WindowSize
    $windowSize.Width = 140
    $windowSize.Height = 38
    $Host.UI.RawUI.WindowSize = $windowSize
} catch { }
Clear-Host
Write-Host "PS $projectRoot> .\.venv\Scripts\python.exe scripts\show_task12.py --section $Section"
& "$projectRoot\.venv\Scripts\python.exe" "$PSScriptRoot\show_task12.py" --section $Section
Write-Host "`nExit code: $LASTEXITCODE"
