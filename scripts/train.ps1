[CmdletBinding()]
param(
    [string]$Dataset = "examples/data/tiny-corpus.txt",
    [string]$Output = "checkpoints/apex-tiny-run",
    [ValidateRange(1, 1000)]
    [int]$Steps = 2,
    [string]$DatasetLicense = "project-authored"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $RepoRoot
& $Python -m apex.cli train-small --dataset $Dataset --output $Output --steps $Steps --dataset-license $DatasetLicense
if ($LASTEXITCODE -ne 0) { throw "Experimental tiny-model training failed." }
