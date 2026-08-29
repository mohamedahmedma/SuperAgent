param(
    [ValidateSet('menu','start','stop','restart','status','logs','rebuild')]
    [string]$Action = 'menu'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$compose = @('compose', '--env-file', '.env', '--env-file', '.runtime.env')

function Ensure-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker Desktop is not installed or docker.exe is not on PATH.'
    }
    docker info *> $null
    if ($LASTEXITCODE -eq 0) { return }
    $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $desktop)) { throw 'Start Docker Desktop and try again.' }
    Write-Host '[setup] Starting Docker Desktop...'
    Start-Process -FilePath $desktop -WindowStyle Hidden
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        Start-Sleep -Seconds 3
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { return }
    }
    throw 'Docker Desktop did not become ready within three minutes.'
}

function Ensure-Config {
    if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host '[setup] Created .env from .env.example.' }
    & (Join-Path $PSScriptRoot 'ensure-runtime-env.ps1')
}

function Get-ConfigValue([string]$Name, [string]$Default) {
    foreach ($file in @('.runtime.env', '.env')) {
        if (-not (Test-Path -LiteralPath $file)) { continue }
        $match = Get-Content -LiteralPath $file | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } | Select-Object -Last 1
        if ($match) { return ($match -split '=', 2)[1].Trim() }
    }
    return $Default
}

function Assert-HostPortsAvailable {
    $ports = @(
        @{ Name='Frontend'; Var='FRONTEND_HOST_PORT'; Default='3000' },
        @{ Name='Backend'; Var='BACKEND_HOST_PORT'; Default='8000' },
        @{ Name='Identity'; Var='IDENTITY_HOST_PORT'; Default='8200' },
        @{ Name='Records'; Var='RECORDS_HOST_PORT'; Default='8100' },
        @{ Name='SIS'; Var='SIS_HOST_PORT'; Default='8300' },
        @{ Name='Attu'; Var='ATTU_HOST_PORT'; Default='8081' },
        @{ Name='PostgreSQL'; Var='POSTGRES_HOST_PORT'; Default='5432' },
        @{ Name='Redis'; Var='REDIS_HOST_PORT'; Default='6379' },
        @{ Name='MinIO API'; Var='MINIO_API_HOST_PORT'; Default='9000' },
        @{ Name='MinIO console'; Var='MINIO_CONSOLE_HOST_PORT'; Default='9001' },
        @{ Name='Milvus'; Var='MILVUS_HOST_PORT'; Default='19530' },
        @{ Name='Milvus health'; Var='MILVUS_HEALTH_HOST_PORT'; Default='9091' }
    )
    foreach ($item in $ports) {
        $port = [int](Get-ConfigValue $item.Var $item.Default)
        $foreignContainer = docker ps --filter "publish=$port" --format '{{.ID}}' | ForEach-Object {
            $own = docker ps --filter "id=$_" --filter 'label=com.docker.compose.project=superagent' --format '{{.ID}}'
            if (-not $own) { docker inspect --format '{{.Name}}' $_ 2>$null }
        } | Select-Object -First 1
        if ($foreignContainer) {
            throw "$($item.Name) host port $port is already published by Docker container $($foreignContainer.TrimStart('/')). Change $($item.Var); no process was stopped."
        }
        $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($listener) {
            $ownContainer = docker ps --filter "publish=$port" --format '{{.ID}}' | Select-Object -First 1
            if (-not $ownContainer) {
                $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
                $processName = if ($process) { $process.ProcessName } else { 'unknown process' }
                throw "$($item.Name) host port $port is already used by Windows process $processName (PID $($listener.OwningProcess)). Change $($item.Var); no process was stopped."
            }
        }
    }
}

function Invoke-Compose([string[]]$Arguments) {
    & docker @compose @Arguments
    if ($LASTEXITCODE -ne 0) { throw "docker compose failed: $($Arguments -join ' ')" }
}

function Wait-Healthy {
    $deadline = (Get-Date).AddMinutes(8)
    do {
        $rows = @(docker @compose ps --format json | ForEach-Object { $_ | ConvertFrom-Json })
        $bad = @($rows | Where-Object { $_.State -notin @('running') -or ($_.Health -and $_.Health -ne 'healthy') })
        if ($rows.Count -ge 11 -and $bad.Count -eq 0) { return }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)
    Invoke-Compose @('ps')
    docker @compose logs --tail 40
    throw 'Not every service became healthy. See the concise logs above.'
}

function Show-Urls {
    $frontendPort = Get-ConfigValue 'FRONTEND_HOST_PORT' '3000'
    $backendPort = Get-ConfigValue 'BACKEND_HOST_PORT' '8000'
    $sisPort = Get-ConfigValue 'SIS_HOST_PORT' '8300'
    $attuPort = Get-ConfigValue 'ATTU_HOST_PORT' '8081'
    $minioPort = Get-ConfigValue 'MINIO_CONSOLE_HOST_PORT' '9001'
    Write-Host ''
    Write-Host "Frontend:     http://localhost:$frontendPort/"
    Write-Host "Backend docs: http://localhost:$backendPort/docs"
    Write-Host "SIS console:  http://localhost:$sisPort/ui"
    Write-Host "Attu:         http://localhost:$attuPort/"
    Write-Host "MinIO:        http://localhost:$minioPort/"
}

function Start-System([switch]$Build) {
    Ensure-Docker; Ensure-Config
    Invoke-Compose @('config', '--quiet')
    Assert-HostPortsAvailable
    $args = @('up', '-d')
    if ($Build) { $args += '--build' }
    Invoke-Compose $args
    Wait-Healthy
    Invoke-Compose @('ps')
    Show-Urls
}

if ($Action -ne 'menu') {
    switch ($Action) {
        'start'   { Start-System }
        'stop'    { Ensure-Docker; Ensure-Config; Invoke-Compose @('stop'); Write-Host 'Stopped. Named volumes and data were preserved.' }
        'restart' { Ensure-Docker; Ensure-Config; Invoke-Compose @('stop'); Start-System }
        'status'  { Ensure-Docker; Ensure-Config; Invoke-Compose @('ps', '-a') }
        'logs'    { Ensure-Docker; Ensure-Config; Invoke-Compose @('logs', '--tail', '150') }
        'rebuild' { Start-System -Build }
    }
    exit 0
}

while ($true) {
    Clear-Host
    Write-Host '========== SuperAgent Docker Manager =========='
    Write-Host '1. Start SuperAgent'
    Write-Host '2. Stop SuperAgent (preserve data)'
    Write-Host '3. Restart SuperAgent'
    Write-Host '4. Show service status'
    Write-Host '5. View logs'
    Write-Host '6. Rebuild changed services'
    Write-Host '7. Open frontend'
    Write-Host '8. Open backend API documentation'
    Write-Host '9. Open Attu'
    Write-Host '10. Exit'
    $choice = (Read-Host 'Choose').Trim()
    try {
        switch ($choice) {
            '1' { Start-System }
            '2' { Ensure-Docker; Ensure-Config; Invoke-Compose @('stop'); Write-Host 'Stopped. Named volumes and data were preserved.' }
            '3' { Ensure-Docker; Ensure-Config; Invoke-Compose @('stop'); Start-System }
            '4' { Ensure-Docker; Ensure-Config; Invoke-Compose @('ps', '-a') }
            '5' { Ensure-Docker; Ensure-Config; $service=Read-Host 'Service name (blank for all)'; if ($service) { Invoke-Compose @('logs','--tail','150',$service) } else { Invoke-Compose @('logs','--tail','150') } }
            '6' { Start-System -Build }
            '7' { Start-Process "http://localhost:$(Get-ConfigValue 'FRONTEND_HOST_PORT' '3000')/" }
            '8' { Start-Process "http://localhost:$(Get-ConfigValue 'BACKEND_HOST_PORT' '8000')/docs" }
            '9' { Start-Process "http://localhost:$(Get-ConfigValue 'ATTU_HOST_PORT' '8081')/" }
            '10' { exit 0 }
            default { Write-Host 'Invalid choice.' }
        }
    } catch {
        Write-Host "[error] $($_.Exception.Message)" -ForegroundColor Red
    }
    if ($choice -ne '10') { Write-Host ''; pause }
}
