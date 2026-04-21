$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $RepoRoot '.runtime\low_resource_rerun'
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$StdoutLogPath = Join-Path $RuntimeRoot "rerun_$Timestamp.out.log"
$StderrLogPath = Join-Path $RuntimeRoot "rerun_$Timestamp.err.log"

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null

$PythonCandidates = @(
    (Join-Path $RepoRoot '.venv\Scripts\python.exe'),
    'python',
    'py'
)

$PythonBin = $null
$PythonArgs = @()
foreach ($candidate in $PythonCandidates) {
    if ($candidate -like '*.exe' -and (Test-Path $candidate)) {
        $PythonBin = $candidate
        break
    }
    if ($candidate -eq 'python') {
        try {
            & python --version *> $null
            $PythonBin = 'python'
            break
        } catch {}
    }
    if ($candidate -eq 'py') {
        try {
            & py -3 --version *> $null
            $PythonBin = 'py'
            $PythonArgs = @('-3')
            break
        } catch {}
    }
}

if (-not $PythonBin) {
    throw 'No usable local Python interpreter was found for runner/serial_low_resource_rerun.py'
}

$ScriptPath = Join-Path $PSScriptRoot 'serial_low_resource_rerun.py'
$Args = @()
$Args += $PythonArgs
$Args += @($ScriptPath)

$Process = Start-Process -FilePath $PythonBin `
    -ArgumentList $Args `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdoutLogPath `
    -RedirectStandardError $StderrLogPath `
    -PassThru

Write-Output "Started low-resource rerun orchestrator."
Write-Output "PID: $($Process.Id)"
Write-Output "Stdout log: $StdoutLogPath"
Write-Output "Stderr log: $StderrLogPath"
