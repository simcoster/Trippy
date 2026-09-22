# Restore public schema from a local custom-format dump or s3://bucket/key.
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

if ($args.Count -lt 1 -or [string]::IsNullOrWhiteSpace($args[0])) {
    throw "usage: restore.ps1 <file.dump|s3://bucket/key|latest>"
}
$src = $args[0]

$BackupDir = if ($env:TRIPPY_BACKUP_DIR) { $env:TRIPPY_BACKUP_DIR } else { Join-Path $Root 'backups' }
$ComposeFile = if ($env:TRIPPY_COMPOSE_FILE) { $env:TRIPPY_COMPOSE_FILE } else { 'docker-compose.yml' }

$EnvFile = Join-Path $Root '.env'
$Compose = @('compose', '-f', (Join-Path $Root $ComposeFile))
if (Test-Path $EnvFile) {
    $Compose += @('--env-file', $EnvFile)
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $k = $line.Substring(0, $eq).Trim()
        $v = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$k" -Value $v
    }
}

if ($src -eq 'latest') {
    foreach ($key in @('BACKUP_S3_BUCKET', 'AWS_ENDPOINT_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')) {
        $value = [Environment]::GetEnvironmentVariable($key)
        if ([string]::IsNullOrWhiteSpace($value)) {
            throw "$key is unset in .env (needed to list s3://`$BACKUP_S3_BUCKET/postgres/)"
        }
        Set-Item -Path "Env:$key" -Value $value.Trim().TrimEnd("`r")
    }
    if (-not (Test-Path $EnvFile)) {
        throw ".env is required to list object storage"
    }
    $listing = docker run --rm `
        --env-file $EnvFile `
        amazon/aws-cli `
        --endpoint-url $env:AWS_ENDPOINT_URL `
        s3 ls "s3://$($env:BACKUP_S3_BUCKET)/postgres/"
    if ($LASTEXITCODE -ne 0) { throw "s3 ls failed ($LASTEXITCODE)" }
    $name = @($listing) | ForEach-Object { ($_ -split '\s+')[-1] } |
        Where-Object { $_ -match '^trippy-.*\.dump$' } |
        Sort-Object |
        Select-Object -Last 1
    if (-not $name) {
        throw "no trippy-*.dump under s3://$($env:BACKUP_S3_BUCKET)/postgres/"
    }
    $src = "s3://$($env:BACKUP_S3_BUCKET)/postgres/$name"
    Write-Host "latest dump $src"
}

$original = $src
$cleanupTmp = $null
if ($src.StartsWith('s3://')) {
    if ([string]::IsNullOrWhiteSpace($env:AWS_ENDPOINT_URL)) {
        throw "AWS_ENDPOINT_URL is required to download"
    }
    if (-not (Test-Path $EnvFile)) {
        throw ".env is required to download from object storage"
    }
    New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
    $tmp = Join-Path $BackupDir 'restore-download.dump'
    $backupAbs = (Resolve-Path $BackupDir).Path
    docker run --rm `
        --env-file $EnvFile `
        -v "${backupAbs}:/data" `
        amazon/aws-cli `
        --endpoint-url $env:AWS_ENDPOINT_URL `
        s3 cp $src /data/restore-download.dump
    if ($LASTEXITCODE -ne 0) { throw "s3 download failed ($LASTEXITCODE)" }
    $src = $tmp
    $cleanupTmp = $tmp
}

if (-not (Test-Path $src)) {
    throw "restore.ps1: not a file: $src"
}
$srcAbs = (Resolve-Path $src).Path

& docker @Compose up -d --wait db
if ($LASTEXITCODE -ne 0) { throw "docker compose up db failed ($LASTEXITCODE)" }
# experiments copies public sequences (nextval defaults). --clean cannot drop
# those sequences while the schema is there. It is a disposable copy.
& docker @Compose exec -T db psql -U trippy -d trippy -v ON_ERROR_STOP=1 `
    -c "DROP SCHEMA IF EXISTS experiments CASCADE"
if ($LASTEXITCODE -ne 0) { throw "drop experiments failed ($LASTEXITCODE)" }

& docker @Compose cp $srcAbs db:/tmp/trippy-restore.dump
if ($LASTEXITCODE -ne 0) { throw "docker compose cp failed ($LASTEXITCODE)" }
& docker @Compose exec -T db pg_restore -U trippy -d trippy --clean --if-exists `
    --exit-on-error -n public /tmp/trippy-restore.dump
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed ($LASTEXITCODE)" }
& docker @Compose exec -T db rm -f /tmp/trippy-restore.dump | Out-Null
if ($cleanupTmp) { Remove-Item -Force $cleanupTmp }
Write-Host "restored public schema from $original"
