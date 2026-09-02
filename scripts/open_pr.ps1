<#
    Push the current branch, open a PR into the base branch, then switch back
    to it. Driven by `just pr`; kept out of the justfile so it can be read,
    diffed and run on its own.

        pwsh scripts/open_pr.ps1
        pwsh scripts/open_pr.ps1 -Title "Split room and site amenities"
#>
[CmdletBinding()]
param(
    # PR title. Empty means derive it from the commits (gh --fill).
    [string] $Title = '',
    [string] $Base = 'main'
)

$ErrorActionPreference = 'Stop'

# $ErrorActionPreference does not stop a failing native exe, so check the code.
function Assert-LastExit([string] $Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
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

# `gh pr list` exits 0 with empty output when there is no open PR; `gh pr view`
# errors instead, which trips $ErrorActionPreference.
$url = gh pr list --head $branch --base $Base --state open --json url --jq '.[0].url'
Assert-LastExit 'Could not query existing pull requests'

if ([string]::IsNullOrWhiteSpace($url)) {
    $ghArgs = @('pr', 'create', '--base', $Base, '--head', $branch, '--fill')
    if (-not [string]::IsNullOrWhiteSpace($Title)) {
        $ghArgs += @('--title', $Title)
    }
    & gh @ghArgs
    Assert-LastExit 'gh pr create failed'
} else {
    Write-Host "PR already open: $url"
}

git checkout $Base
Assert-LastExit "Could not check out $Base"

# The PR is not merged yet, so this only freshens $Base for the next branch.
git pull --ff-only
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not fast-forward $Base - pull it manually."
}
exit 0
