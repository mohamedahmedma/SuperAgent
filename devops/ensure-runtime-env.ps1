$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $root '.runtime.env'
$envFile = Join-Path $root '.env'

function New-Secret([int]$Bytes = 32) {
    $buffer = New-Object byte[] $Bytes
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($buffer) } finally { $rng.Dispose() }
    return ([BitConverter]::ToString($buffer) -replace '-', '').ToLowerInvariant()
}

if (-not (Test-Path -LiteralPath $runtime)) {
    $content = @(
        '# Generated locally by SUPERAGENT.bat. Never commit this file.'
        'POSTGRES_PASSWORD=' + (New-Secret 24)
        'MINIO_ROOT_USER=superagent'
        'MINIO_ROOT_PASSWORD=' + (New-Secret 24)
        'LOCAL_SERVICE_KEY=' + (New-Secret 32)
    )
    [IO.File]::WriteAllLines($runtime, $content, [Text.UTF8Encoding]::new($false))
    Write-Host '[setup] Created ignored .runtime.env with local-only random credentials.'
}

# Docker Desktop and a plain `docker compose up` only interpolate `.env`; they do not
# know about the CLI's additional `--env-file .runtime.env`. Keep the ignored `.env`
# synchronized so both launch paths produce exactly the same container configuration.
$runtimeValues = @{}
Get-Content -LiteralPath $runtime | ForEach-Object {
    if ($_ -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
        $runtimeValues[$matches[1]] = $matches[2]
    }
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $root '.env.example') -Destination $envFile
}

$envLines = [Collections.Generic.List[string]]::new()
Get-Content -LiteralPath $envFile | ForEach-Object { $envLines.Add($_) }
foreach ($name in @('POSTGRES_PASSWORD', 'MINIO_ROOT_USER', 'MINIO_ROOT_PASSWORD', 'LOCAL_SERVICE_KEY')) {
    if (-not $runtimeValues.ContainsKey($name)) { throw "$name is missing from .runtime.env" }
    $replacement = "$name=$($runtimeValues[$name])"
    $index = -1
    for ($i = 0; $i -lt $envLines.Count; $i++) {
        if ($envLines[$i] -match "^$([regex]::Escape($name))=") { $index = $i; break }
    }
    if ($index -ge 0) { $envLines[$index] = $replacement } else { $envLines.Add($replacement) }
}
[IO.File]::WriteAllLines($envFile, $envLines, [Text.UTF8Encoding]::new($false))
