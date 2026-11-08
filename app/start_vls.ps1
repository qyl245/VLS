[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Foreground,
    [switch]$NoBrowser,
    [switch]$RebuildFrontend,
    [switch]$Setup
)

$ErrorActionPreference = 'Stop'
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $AppRoot '.venv\Scripts\python.exe'
$VenvDir = Join-Path $AppRoot '.venv'
$AnacondaPython = 'D:\Anaconda\python.exe'
$FrontendDir = Join-Path $AppRoot 'web\frontend'
$DistIndex = Join-Path $FrontendDir 'dist\index.html'
$NodeModules = Join-Path $FrontendDir 'node_modules'
$WebDataDir = Join-Path $AppRoot 'webdata'
$PidFile = Join-Path $WebDataDir 'vls.pid'
$StdoutLog = Join-Path $WebDataDir 'vls.stdout.log'
$StderrLog = Join-Path $WebDataDir 'vls.stderr.log'
$Port = 8011
$BaseUrl = "http://127.0.0.1:$Port"

Set-Location $AppRoot

function Invoke-Checked {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

function Test-PythonDependencies {
    if (-not (Test-Path -LiteralPath $Python)) { return $false }
    & $Python -c "import fastapi,uvicorn,cv2,mediapipe,numpy,scipy,jsonschema" *> $null
    return $LASTEXITCODE -eq 0
}

function Test-VlsHealth {
    try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/api/v1/health" -TimeoutSec 2
        return $health.status -eq 'ok'
    }
    catch {
        return $false
    }
}

function Get-PortOwner {
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($connection) { return [int]$connection.OwningProcess }
    return $null
}

function Test-VlsProcess {
    param([int]$ProcessId)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $process -or -not $process.CommandLine) { return $false }
    return $process.CommandLine -match 'uvicorn' -and $process.CommandLine -match 'app\.main:app'
}

function Stop-Vls {
    $targetIds = @()
    if (Test-Path -LiteralPath $PidFile) {
        $savedPid = 0
        [void][int]::TryParse((Get-Content -LiteralPath $PidFile -Raw).Trim(), [ref]$savedPid)
        if ($savedPid -gt 0 -and (Test-VlsProcess -ProcessId $savedPid)) { $targetIds += $savedPid }
    }
    $portOwner = Get-PortOwner
    if ($portOwner -and (Test-VlsProcess -ProcessId $portOwner)) { $targetIds += [int]$portOwner }
    $targetIds = @($targetIds | Select-Object -Unique)

    if ($targetIds.Count -eq 0) {
        if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
        Write-Host 'VLS is not running.'
        return
    }
    foreach ($targetId in $targetIds) {
        Write-Host "Stopping VLS process $targetId..."
        Stop-Process -Id $targetId -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds(8)
    while ((Get-Date) -lt $deadline -and (Test-VlsHealth)) { Start-Sleep -Milliseconds 300 }
    foreach ($targetId in $targetIds) {
        if (Get-Process -Id $targetId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $targetId -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
    Write-Host 'VLS stopped.'
}

function Ensure-Frontend {
    if (-not (Test-Path -LiteralPath $NodeModules) -or $Setup) {
        $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
        Write-Host 'Installing frontend dependencies...'
        Push-Location $FrontendDir
        try {
            Invoke-Checked -FilePath $npm -Arguments @('ci', '--cache', '.npm-cache')
        }
        finally {
            Pop-Location
        }
    }

    $needsBuild = $RebuildFrontend -or $Setup -or -not (Test-Path -LiteralPath $DistIndex)
    if (-not $needsBuild -and (Test-Path -LiteralPath $DistIndex)) {
        $distTime = (Get-Item -LiteralPath $DistIndex).LastWriteTimeUtc
        $sourceTime = Get-ChildItem -LiteralPath (Join-Path $FrontendDir 'src') -File -Recurse |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        $configTime = Get-ChildItem -LiteralPath $FrontendDir -File |
            Where-Object { $_.Name -in @('package.json', 'package-lock.json', 'vite.config.ts', 'tsconfig.json') } |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if (($sourceTime -and $sourceTime.LastWriteTimeUtc -gt $distTime) -or
            ($configTime -and $configTime.LastWriteTimeUtc -gt $distTime)) {
            $needsBuild = $true
        }
    }

    if ($needsBuild) {
        $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
        Write-Host 'Building frontend...'
        Push-Location $FrontendDir
        try {
            Invoke-Checked -FilePath $npm -Arguments @('run', 'build')
        }
        finally {
            Pop-Location
        }
    }
}

function Ensure-Python {
    if (-not (Test-Path -LiteralPath $Python)) {
        if (-not (Test-Path -LiteralPath $AnacondaPython)) {
            throw "Python environment is missing. Expected Anaconda at $AnacondaPython."
        }
        Write-Host 'Creating Python virtual environment from Anaconda...'
        Invoke-Checked -FilePath $AnacondaPython -Arguments @('-m', 'venv', $VenvDir)
    }

    if ($Setup -or -not (Test-PythonDependencies)) {
        Write-Host 'Installing backend dependencies...'
        Invoke-Checked -FilePath $Python -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Checked -FilePath $Python -Arguments @(
            '-m', 'pip', 'install',
            'jax==0.4.38', 'jaxlib==0.4.38',
            '-r', 'requirements.txt'
        )
    }
}

if ($Stop) {
    Stop-Vls
    exit 0
}

if (-not (Test-Path -LiteralPath $WebDataDir)) {
    New-Item -ItemType Directory -Path $WebDataDir | Out-Null
}

if (Test-VlsHealth) {
    $owner = Get-PortOwner
    if ($owner) {
        Set-Content -LiteralPath $PidFile -Value $owner -Encoding ASCII
        Write-Host "VLS is already running at $BaseUrl (PID $owner)."
    }
    else {
        Write-Host "VLS is already running at $BaseUrl."
    }
    if (-not $NoBrowser) { Start-Process $BaseUrl }
    exit 0
}

$portOwner = Get-PortOwner
if ($portOwner) {
    throw "Port $Port is already used by PID $portOwner, but it is not the VLS service."
}

Ensure-Python
Ensure-Frontend

$env:VLS_PYTHON_EXE = $Python
$env:VLS_WEBDATA_ROOT = $WebDataDir
$uvicornArgs = @(
    '-m', 'uvicorn', 'app.main:app',
    '--app-dir', 'web\backend',
    '--host', '127.0.0.1',
    '--port', "$Port"
)

if ($Foreground) {
    Write-Host "Starting VLS at $BaseUrl. Press Ctrl+C to stop."
    & $Python @uvicornArgs
    exit $LASTEXITCODE
}

Write-Host 'Starting VLS...'
$process = Start-Process -FilePath $Python -ArgumentList $uvicornArgs -WorkingDirectory $AppRoot -WindowStyle Hidden -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ASCII

$deadline = (Get-Date).AddSeconds(60)
do {
    Start-Sleep -Milliseconds 500
    $process.Refresh()
    if ($process.HasExited) { break }
} while ((Get-Date) -lt $deadline -and -not (Test-VlsHealth))

if (-not (Test-VlsHealth)) {
    if (Test-Path -LiteralPath $StderrLog) {
        Write-Host '--- stderr ---'
        Get-Content -LiteralPath $StderrLog -Tail 40
    }
    if (Test-Path -LiteralPath $StdoutLog) {
        Write-Host '--- stdout ---'
        Get-Content -LiteralPath $StdoutLog -Tail 40
    }
    throw "VLS failed to start. See $StdoutLog and $StderrLog."
}

Write-Host "VLS started at $BaseUrl (PID $($process.Id))."
Write-Host "Logs: $StdoutLog and $StderrLog"
if (-not $NoBrowser) { Start-Process $BaseUrl }
