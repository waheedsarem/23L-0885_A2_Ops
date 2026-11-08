$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$dvc = Join-Path $projectRoot '.venv\Scripts\dvc.exe'
$env:DVC_SITE_CACHE_DIR = Join-Path $projectRoot '.dvc\site-cache'

Write-Host 'A Windows credential dialog will open.'
Write-Host 'Keep username waheedsarem and paste the DagsHub access token into the password field.'
Write-Host 'The token is stored only in Git-ignored .dvc/config.local.'
$credential = Get-Credential -UserName 'waheedsarem' -Message 'Paste your DagsHub access token into the password field'
if ($null -eq $credential) { throw 'Credential entry was cancelled.' }
$secureToken = $credential.Password
$tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
    if ([string]::IsNullOrWhiteSpace($token)) { throw 'No token was entered.' }
    & $dvc remote modify origin --local access_key_id $token
    if ($LASTEXITCODE -ne 0) { throw 'Failed to set DVC access key.' }
    & $dvc remote modify origin --local secret_access_key $token
    if ($LASTEXITCODE -ne 0) { throw 'Failed to set DVC secret key.' }
    Write-Host 'DVC credentials configured in ignored local configuration.'
} finally {
    if ($tokenPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    }
    $token = $null
    $secureToken = $null
    $credential = $null
}
