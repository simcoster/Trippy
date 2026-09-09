# Run from the repo root. Requires just (https://github.com/casey/just) + uv.
# Pipeline: scrape-sites → scrape-info (rooms → prices → rules → breadcrumbs) → scrape-availability

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Both, and the repo root first. `source/scraper` is what lets the scrapers'
# bare `from amenity_enrichment import ...` / `from info_site import ...`
# imports resolve; the repo root is what lets those same modules reach `db.models`
# and `source.scraper.*`. Without the root, running a scraper as a script put
# only `source/scraper` on the path and `just scrape-availability` died at
# import with "No module named 'db'" — a script's sys.path[0] is its own
# directory, never the working directory.
path_sep := if os_family() == "windows" { ";" } else { ":" }
export PYTHONPATH := justfile_directory() + path_sep + justfile_directory() / "source/scraper"

[private]
default:
    @just --list

# Slugify a title, check out that branch, push to origin
[windows]
branch name:
    #!powershell
    $ErrorActionPreference = 'Stop'
    $raw = {{ quote(name) }}
    $branch = ($raw.ToLowerInvariant() -replace '[^a-z0-9/_-]+', '-' -replace '-{2,}', '-').Trim('-').Trim('/')
    if ([string]::IsNullOrWhiteSpace($branch)) {
        throw "Could not make a git branch name from: $raw"
    }
    git check-ref-format --branch $branch
    if ($LASTEXITCODE -ne 0) { throw "Invalid branch name: $branch" }
    git checkout -b $branch
    git push -u origin $branch

# Slugify a title, check out that branch, push to origin
[unix]
branch name:
    #!/usr/bin/env bash
    set -euo pipefail
    raw={{ quote(name) }}
    branch="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9/_-]+#-#g; s#-{2,}#-#g; s#^-+##; s#-+$##; s#^/+##; s#/+$##')"
    [[ -n "$branch" ]] || { echo "Could not make a git branch name from: $raw" >&2; exit 1; }
    git check-ref-format --branch "$branch" >/dev/null || { echo "Invalid branch name: $branch" >&2; exit 1; }
    git checkout -b "$branch"
    git push -u origin "$branch"

# Push the current branch, open a PR into main, wait for CI; merge if green
[windows]
pr *title:
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/open_pr.ps1 {{ if title == "" { "" } else { "-Title " + quote(title) } }}

# Push the current branch, open a PR into main, wait for CI; merge if green
[unix]
pr *title:
    bash scripts/open_pr.sh {{ if title == "" { "" } else { quote(title) } }}

# Copy public → experiments. `just setup-experiments` or `… copy --empty t1,t2`
setup-experiments *args:
    uv run python scripts/setup_experiments.py {{ args }}

# Extractor + planner on evals/planner_v1.json (LLM; frozen occupancy).
# Copies public → experiments first, except availability.
run-eval *args:
    uv run python -m source.eval.run {{ trim_start_match(args, "-- ") }}

# Run any just recipe with TRIPPY_SCHEMA=experiments (scrapes, planner, clears)
#   just on-experiments scrape-breadcrumbs
#   just on-experiments scrape-breadcrumbs -- --site 5
[windows]
on-experiments +args:
    $env:TRIPPY_SCHEMA = "experiments"; just {{ args }}

[unix]
on-experiments +args:
    TRIPPY_SCHEMA=experiments just {{ args }}

# info-site lodging panel → accommodation_types + per-unit rules (--site N)
scrape-rooms *args:
    uv run python -m source.scraper.rules_ingest.rooms {{ trim_start_match(args, "-- ") }}

# All info-page scrapes in dependency order: rooms → prices → rules → breadcrumbs
scrape-info *args:
    just scrape-rooms {{ args }}
    just scrape-prices {{ args }}
    just scrape-rules {{ args }}
    just scrape-breadcrumbs {{ args }}

# parks.org.il #breadcrumbs → region claims (--site N)
scrape-breadcrumbs *args:
    uv run python -m source.scraper.info_site.breadcrumbs {{ trim_start_match(args, "-- ") }}

# info-site rate cards → list_prices (--site N)
scrape-prices *args:
    uv run python -m source.scraper.info_site.scrape --prices {{ trim_start_match(args, "-- ") }}

# parks.org.il listing → campsites
scrape-sites:
    uv run python source/scraper/discover_sites.py

# INPA vacancies → availability (match existing types). One site: -- --site 2
scrape-availability *args:
    uv run python source/scraper/populate_availability.py {{ trim_start_match(args, "-- ") }}

# Google Place Details → reviews table only (newest then most_relevant, concat)
scrape-reviews *args:
    uv run python -m source.scraper.populate_reviews_and_claims {{ trim_start_match(args, "-- ") }}

# Visit-gate + split claims for reviews with is_relevant IS NULL (--campsite-id N)
populate-claims *args:
    uv run python -m source.scraper.populate_claims {{ trim_start_match(args, "-- ") }}

# info-site static pages -> campsite_rules (site-level rules + amenities; --site N)
scrape-rules *args:
    uv run python -m source.scraper.rules_ingest.ingest {{ trim_start_match(args, "-- ") }}

# Delete availability rows; keeps types, rules and prices (--types, --site N)
clear-availability *args:
    uv run python scripts/clear_availability.py {{ trim_start_match(args, "-- ") }}

# Truncate Google reviews; keep campsites and breadcrumb region claims
clear-reviews:
    uv run python scripts/clear_reviews_and_claims.py

# Delete claims and null reviews.is_relevant; keep review rows
clear-claims:
    uv run python scripts/clear_claims.py

# Clear all info-page data: rules, prices, types, names, vocabulary, breadcrumb claims + availability
clear-info *args:
    uv run python scripts/clear_info.py {{ trim_start_match(args, "-- ") }}

# Delete site-level campsite_rules; keeps per-unit rows + vocabulary (--all, --subjects, --site N)
clear-rules *args:
    uv run python scripts/clear_rules.py {{ trim_start_match(args, "-- ") }}

# Apply pending Alembic migrations
update-tables:
    uv run alembic upgrade head

# sites, then everything the info page gives, then availability
scrape-all: scrape-sites scrape-info scrape-availability

# Local Streamlit agent (Telegram remains production)
streamlit:
    uv run streamlit run scripts/streamlit_chat.py
