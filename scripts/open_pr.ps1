<#
    Push the current branch, open a PR into the base branch, wait for CI.
    Stays on the feature branch. Driven by `just pr`; kept out of the justfile
    so it can be read, diffed and run on its own.

        pwsh scripts/open_pr.ps1
        pwsh scripts/open_pr.ps1 -Title "Split room and site amenities"
#>
[CmdletBinding()]
param(
    # PR title. Empty means humanize the branch name (hyphens to spaces).
    [string] $Title = '',
    [string] $Base = 'main'
)

$ErrorActionPreference = 'Stop'

# $ErrorActionPreference does not stop a failing native exe, so check the code.
function Assert-LastExit([string] $Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}

function ConvertTo-TitleFromBranch([string] $Name) {
    $t = ($Name -replace '[-_]+', ' ' -replace '/', ': ').Trim()
    if ([string]::IsNullOrWhiteSpace($t)) { return $Name }
    return $t.Substring(0, 1).ToUpperInvariant() + $t.Substring(1)
}

function Wait-ChecksReported([string] $PrUrl) {
    for ($i = 0; $i -lt 24; $i++) {
        $json = gh pr checks $PrUrl --json name 2>$null
        if ($json -and $json.Trim() -ne '[]') { return $true }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Show-FailedCi([string] $PrUrl, [string] $Branch) {
    Write-Host ''
    Write-Host 'CI failed.'
    gh pr checks $PrUrl
    $sha = (git rev-parse HEAD).Trim()
    $runsJson = gh run list --branch $Branch --limit 10 --json databaseId,conclusion,headSha,name,url
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($runsJson)) {
        Write-Host 'Could not list GitHub Actions runs. See the checks above.'
        return
    }
    $failed = @(
        $runsJson | ConvertFrom-Json |
            Where-Object { $_.headSha -eq $sha -and $_.conclusion -eq 'failure' }
    )
    if ($failed.Count -eq 0) {
        Write-Host 'No failed GitHub Actions runs found for this commit. See the checks above.'
        return
    }
    foreach ($run in $failed) {
        Write-Host ''
        Write-Host "----- $($run.name) $($run.url) -----"
        gh run view $run.databaseId --log-failed
    }
}

$branch = git rev-parse --abbrev-ref HEAD
Assert-LastExit 'Not a git repository'
$branch = $branch.Trim()

if ($branch -eq $Base -or $branch -eq 'HEAD') {
    throw "Already on $branch - start a feature branch first: just branch ""My title"""
}
if (git status --porcelain) {
    throw 'Working tree is dirty - commit or stash before opening a PR'
}

git fetch origin $Base
Assert-LastExit "Could not fetch origin/$Base"

$ahead = git rev-list --count "origin/$Base..HEAD"
Assert-LastExit "Could not compare $branch with origin/$Base"
if ($ahead.Trim() -eq '0') {
    throw "$branch has no commits that $Base does not already have"
}

git push -u origin $branch
Assert-LastExit "Could not push $branch"

if ([string]::IsNullOrWhiteSpace($Title)) {
    $Title = ConvertTo-TitleFromBranch $branch
}

# `gh pr list` exits 0 with empty output when there is no open PR; `gh pr view`
# errors instead, which trips $ErrorActionPreference.
$url = gh pr list --head $branch --base $Base --state open --json url --jq '.[0].url'
Assert-LastExit 'Could not query existing pull requests'

if ([string]::IsNullOrWhiteSpace($url)) {
    Write-Host "Opening PR: $Title"
    $ghArgs = @('pr', 'create', '--base', $Base, '--head', $branch, '--fill', '--title', $Title)
    & gh @ghArgs
    Assert-LastExit 'gh pr create failed'
} else {
    Write-Host "PR already open: $url"
}

$url = gh pr view --json url --jq .url
Assert-LastExit 'Could not get PR URL'
$url = $url.Trim()
Write-Host "PR: $url"

if (-not (Wait-ChecksReported $url)) {
    Write-Host "No CI checks reported within 2 minutes. PR is open: $url"
    exit 0
}

Write-Host 'Waiting for CI to finish...'
gh pr checks $url --watch
if ($LASTEXITCODE -eq 0) {
    Write-Host 'CI passed.'
    Write-Host $url
    exit 0
}

Show-FailedCi $url $branch
exit 1
