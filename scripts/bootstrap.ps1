$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repo = "https://github.com/Snowflake-Labs/dare-bench.git"
$Commit = "01447145304c67b861a004ada6d86f29640de61a"
$Vendor = "vendor\dare-bench"

function Invoke-Native {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Force -Path "vendor" | Out-Null

if (Test-Path $Vendor) {
    if (-not (Test-Path "$Vendor\.git")) {
        Write-Host "Removing incomplete DARE-Bench download..."
        Remove-Item -Recurse -Force $Vendor
    }
}

if (-not (Test-Path "$Vendor\.git")) {
    New-Item -ItemType Directory -Force -Path $Vendor | Out-Null
    Invoke-Native git -C $Vendor init
    Invoke-Native git -C $Vendor remote add origin $Repo
}

Write-Host "Fetching pinned DARE-Bench revision..."
Invoke-Native git -c http.version=HTTP/1.1 -C $Vendor fetch --depth 1 origin $Commit
Invoke-Native git -C $Vendor checkout --detach FETCH_HEAD

$ActualCommit = (& git -C $Vendor rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not read DARE-Bench HEAD."
}
if ($ActualCommit -ne $Commit) {
    throw "Wrong DARE-Bench revision. Expected $Commit but got $ActualCommit."
}

if (-not (Test-Path "$Vendor\data\eval\question_list.json")) {
    throw "DARE-Bench checkout is incomplete: data\eval\question_list.json is missing."
}

$PythonArgs = @()
$Py312 = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
if ($LASTEXITCODE -eq 0 -and $Py312) {
    $PythonArgs = @("-3.12")
    Write-Host "Using Python 3.12."
}
else {
    Write-Warning "Python 3.12 was not found. Falling back to the default 'py' interpreter."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($PythonArgs.Count -gt 0) {
        Invoke-Native py @PythonArgs -m venv .venv
    }
    else {
        Invoke-Native py -m venv .venv
    }
}

Invoke-Native ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
Invoke-Native ".\.venv\Scripts\python.exe" -m pip install -r requirements-analysis.txt

$PythonVersion = & ".\.venv\Scripts\python.exe" -c "import sys; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "Could not verify Python environment."
}

Write-Host ""
Write-Host "Bootstrap verified successfully."
Write-Host "DARE-Bench commit: $ActualCommit"
Write-Host "Python: $PythonVersion"
Write-Host "Question list: $Vendor\data\eval\question_list.json"
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
