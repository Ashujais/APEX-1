[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $RepoRoot
& $Python scripts\detect_hardware.py
if ($LASTEXITCODE -ne 0) { throw "Hardware detection failed." }
