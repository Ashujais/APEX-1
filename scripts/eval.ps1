[CmdletBinding()]
param(
    [string]$Checkpoint = "checkpoints/apex-tiny-run/checkpoint-final.pt",
    [string]$Tokenizer = "checkpoints/apex-tiny-run/tokenizer.json",
    [string]$Dataset = "examples/data/tiny-corpus.txt"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $RepoRoot
& $Python -m apex.cli evaluate --checkpoint $Checkpoint --tokenizer $Tokenizer --dataset $Dataset
if ($LASTEXITCODE -ne 0) { throw "Tiny-model evaluation failed." }
