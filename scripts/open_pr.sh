#!/usr/bin/env bash
# Push the current branch, open a PR into the base branch, then switch back
# to it. Driven by `just pr` on macOS/Linux; the PowerShell twin is
# open_pr.ps1 and the two must stay in step.
#
#     scripts/open_pr.sh
#     scripts/open_pr.sh "Split room and site amenities"
#     scripts/open_pr.sh "Title" develop        # different base branch
set -euo pipefail

title="${1:-}"      # empty means derive it from the commits (gh --fill)
base="${2:-main}"

die() { echo "$*" >&2; exit 1; }

branch="$(git rev-parse --abbrev-ref HEAD)" || die "Not a git repository"
if [[ "$branch" == "$base" || "$branch" == "HEAD" ]]; then
    die "Already on $branch - start a feature branch first: just branch \"My title\""
fi
if [[ -n "$(git status --porcelain)" ]]; then
    die "Working tree is dirty - commit or stash before opening a PR"
fi

git fetch origin "$base" || die "Could not fetch origin/$base"

ahead="$(git rev-list --count "origin/$base..HEAD")" || die "Could not compare $branch with origin/$base"
if [[ "$ahead" == "0" ]]; then
    die "$branch has no commits that $base does not already have"
fi

git push -u origin "$branch" || die "Could not push $branch"

# `gh pr list` exits 0 with empty output when there is no open PR; `gh pr view`
# errors instead.
url="$(gh pr list --head "$branch" --base "$base" --state open --json url --jq '.[0].url')" \
    || die "Could not query existing pull requests"

if [[ -z "$url" ]]; then
    args=(pr create --base "$base" --head "$branch" --fill)
    if [[ -n "$title" ]]; then
        args+=(--title "$title")
    fi
    gh "${args[@]}" || die "gh pr create failed"
else
    echo "PR already open: $url"
fi

git checkout "$base" || die "Could not check out $base"

# The PR is not merged yet, so this only freshens $base for the next branch.
if ! git pull --ff-only; then
    echo "warning: could not fast-forward $base - pull it manually." >&2
fi
