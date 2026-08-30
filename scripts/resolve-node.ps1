function Resolve-ApexNodeRuntime {
    $nodePath = $env:APEX_NODE_EXECUTABLE
    if (-not $nodePath) {
        $nodeCommand = Get-Command node.exe -ErrorAction Stop
        $nodePath = $nodeCommand.Source
    }
    if (-not (Test-Path -LiteralPath $nodePath -PathType Leaf)) {
        throw "Node executable not found at $nodePath"
    }
    $nodeVersion = [version]((& $nodePath --version).Trim().TrimStart("v"))
    if ($nodeVersion -lt [version]"22.13.0") {
        throw "APEX web requires Node >=22.13.0; found $nodeVersion at $nodePath"
    }

    $npmCli = $env:APEX_NPM_CLI
    if (-not $npmCli) {
        $npmCli = Join-Path (Split-Path -Parent $nodePath) "node_modules\npm\bin\npm-cli.js"
    }
    if (-not (Test-Path -LiteralPath $npmCli -PathType Leaf)) {
        throw "npm CLI not found for $nodePath; set APEX_NPM_CLI to npm-cli.js"
    }
    [pscustomobject]@{
        Node = $nodePath
        NpmCli = $npmCli
        Version = $nodeVersion
    }
}
