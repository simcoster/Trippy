# Dump public schema to backups/. Docker compose cp, not `>` (corrupts -Fc).
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

$BackupDir = if ($env:TRIPPY_BACKUP_DIR) { $env:TRIPPY_BACKUP_DIR } else { Join-Path $Root 'backups' }
$ComposeFile = if ($env:TRIPPY_COMPOSE_FILE) { $env:TRIPPY_COMPOSE_FILE } else { 'docker-compose.yml' }
$KeepLocal = if ($env:TRIPPY_BACKUP_KEEP_LOCAL) { [int]$env:TRIPPY_BACKUP_KEEP_LOCAL } else { 7 }

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

New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$name = "trippy-$stamp.dump"
$remote = "/tmp/$name"
$dump = Join-Path $BackupDir $name

& docker @Compose exec -T db pg_dump -U trippy -n public -Fc -f $remote trippy
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed ($LASTEXITCODE)" }
& docker @Compose cp "db:$remote" $dump
if ($LASTEXITCODE -ne 0) { throw "docker compose cp failed ($LASTEXITCODE)" }
& docker @Compose exec -T db rm -f $remote | Out-Null

$bytes = (Get-Item $dump).Length
Write-Host "wrote $dump ($bytes bytes, public schema)"

Get-ChildItem -Path $BackupDir -Filter 'trippy-*.dump' -File -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTimeUtc -lt [DateTime]::UtcNow.AddDays(-$KeepLocal) } |
    Remove-Item -Force

if (-not [string]::IsNullOrWhiteSpace($env:BACKUP_S3_BUCKET)) {
    if ([string]::IsNullOrWhiteSpace($env:AWS_ENDPOINT_URL)) {
        throw "AWS_ENDPOINT_URL is required to upload"
    }
    if (-not (Test-Path $EnvFile)) {
        throw ".env is required to upload to object storage"
    }
    $dumpAbs = (Resolve-Path $dump).Path
    docker run --rm `
        --env-file $EnvFile `
        -v "${dumpAbs}:/data/dump:ro" `
        amazon/aws-cli `
        --endpoint-url $env:AWS_ENDPOINT_URL `
        s3 cp /data/dump "s3://$($env:BACKUP_S3_BUCKET)/postgres/$name"
    if ($LASTEXITCODE -ne 0) { throw "s3 upload failed ($LASTEXITCODE)" }
    Write-Host "uploaded s3://$($env:BACKUP_S3_BUCKET)/postgres/$name"
}
