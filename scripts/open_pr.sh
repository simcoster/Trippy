#!/usr/bin/env bash
# Push the current branch, open a PR into the base branch, wait for CI.
# Stays on the feature branch. Driven by `just pr` on macOS/Linux; the
# PowerShell twin is open_pr.ps1 and the two must stay in step.
#
#     scripts/open_pr.sh
#     scripts/open_pr.sh "Split room and site amenities"
#     scripts/open_pr.sh "Title" develop        # different base branch
set -euo pipefail

title="${1:-}"      # empty means humanize the branch name (hyphens to spaces)
base="${2:-main}"

die() { echo "$*" >&2; exit 1; }

title_from_branch() {
    local b="$1"
    b="${b//-/ }"
    b="${b//_/ }"
    b="${b//\//: }"
    printf '%s%s' "$(printf '%s' "$b" | cut -c1 | tr '[:lower:]' '[:upper:]')" "$(printf '%s' "$b" | cut -c2-)"
}

wait_checks_reported() {
    local pr_url="$1" i json
    for i in $(seq 1 24); do
        json="$(gh pr checks "$pr_url" --json name 2>/dev/null || true)"
        if [[ -n "$json" && "$json" != "[]" ]]; then
            return 0
        fi
        sleep 5
    done
    return 1
}

show_failed_ci() {
    local pr_url="$1" branch="$2" sha ids id
    echo
    echo "CI failed."
    gh pr checks "$pr_url" || true
    sha="$(git rev-parse HEAD)"
    ids="$(gh run list --branch "$branch" --limit 10 --json databaseId,conclusion,headSha \
        --jq ".[] | select(.headSha == \"$sha\" and .conclusion == \"failure\") | .databaseId" || true)"
    if [[ -z "$ids" ]]; then
        echo "No failed GitHub Actions runs found for this commit. See the checks above."
        return
    fi
    for id in $ids; do
        echo
        echo "----- failed run $id -----"
        gh run view "$id" --log-failed || true
    done
}

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

if [[ -z "$title" ]]; then
    title="$(title_from_branch "$branch")"
fi

# `gh pr list` exits 0 with empty output when there is no open PR; `gh pr view`
# errors instead.
url="$(gh pr list --head "$branch" --base "$base" --state open --json url --jq '.[0].url')" \
    || die "Could not query existing pull requests"

if [[ -z "$url" ]]; then
    echo "Opening PR: $title"
    gh pr create --base "$base" --head "$branch" --fill --title "$title" \
        || die "gh pr create failed"
else
    echo "PR already open: $url"
fi

url="$(gh pr view --json url --jq .url)" || die "Could not get PR URL"
echo "PR: $url"

if ! wait_checks_reported "$url"; then
    echo "No CI checks reported within 2 minutes. PR is open: $url"
    exit 0
fi

echo "Waiting for CI to finish..."
set +e
gh pr checks "$url" --watch
ci_status=$?
set -e

if [[ "$ci_status" -eq 0 ]]; then
    echo "CI passed."
    echo "$url"
    exit 0
fi

show_failed_ci "$url" "$branch"
exit 1
