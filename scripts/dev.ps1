[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$WebRoot = Join-Path $RepoRoot "apps\web"
. (Join-Path $PSScriptRoot "resolve-node.ps1")
$NodeRuntime = Resolve-ApexNodeRuntime
$env:Path = (Split-Path -Parent $NodeRuntime.Node) + ";" + $env:Path

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python virtual environment not found at $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $WebRoot "package-lock.json") -PathType Leaf)) {
    throw "apps/web/package-lock.json is required; this repository uses npm."
}

Write-Host "Starting APEX API on http://localhost:$ApiPort"
Write-Host "Starting APEX web app on http://localhost:$WebPort"
$ApiJob = Start-Job -ScriptBlock {
    param($WorkingDirectory, $PythonPath, $Port)
    Set-Location -LiteralPath $WorkingDirectory
    & $PythonPath -m uvicorn apex_api.main:app --reload --port $Port
} -ArgumentList $RepoRoot, $Python, $ApiPort
$WebJob = Start-Job -ScriptBlock {
    param($WorkingDirectory, $Port, $NodePath, $NpmCli)
    Set-Location -LiteralPath $WorkingDirectory
    $env:Path = (Split-Path -Parent $NodePath) + ";" + $env:Path
    & $NodePath $NpmCli run dev -- --port $Port
} -ArgumentList $WebRoot, $WebPort, $NodeRuntime.Node, $NodeRuntime.NpmCli

try {
    $ActiveStates = @("NotStarted", "Running")
    while ($ApiJob.State -in $ActiveStates -and $WebJob.State -in $ActiveStates) {
        Receive-Job -Job $ApiJob, $WebJob -ErrorAction Continue
        Start-Sleep -Milliseconds 500
    }
    Receive-Job -Job $ApiJob, $WebJob -ErrorAction Continue
    throw "A development process exited; inspect the output above."
}
finally {
    Stop-Job -Job $ApiJob, $WebJob -ErrorAction SilentlyContinue
    Remove-Job -Job $ApiJob, $WebJob -Force -ErrorAction SilentlyContinue
}
