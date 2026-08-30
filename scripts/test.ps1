[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$WebRoot = Join-Path $RepoRoot "apps\web"
. (Join-Path $PSScriptRoot "resolve-node.ps1")
$NodeRuntime = Resolve-ApexNodeRuntime
$env:Path = (Split-Path -Parent $NodeRuntime.Node) + ";" + $env:Path

Set-Location -LiteralPath $RepoRoot
& $Python scripts\run_tests.py
if ($LASTEXITCODE -ne 0) { throw "Python tests failed." }
& $Python -m ruff check .
if ($LASTEXITCODE -ne 0) { throw "Python lint failed." }

Push-Location -LiteralPath $WebRoot
try {
    & $NodeRuntime.Node $NodeRuntime.NpmCli ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
    & $NodeRuntime.Node $NodeRuntime.NpmCli run lint
    if ($LASTEXITCODE -ne 0) { throw "Frontend lint failed." }
    & $NodeRuntime.Node $NodeRuntime.NpmCli run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
}
finally {
    Pop-Location
}
