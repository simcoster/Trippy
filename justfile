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
export PRICE_SANDBOX_URL := env("PRICE_SANDBOX_URL", "http://127.0.0.1:8503")

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

# Copy public → experiments; drops leftover extras. `just setup-experiments` or `… copy --empty t1,t2`
setup-experiments *args:
    uv run python scripts/setup_experiments.py {{ args }}

# Extractor + planner on evals/planner_v1.json (LLM; frozen occupancy).
# Copies public → experiments first, except availability.
# `just run-eval -- --recommender` also dumps 1–2 cited recs (not scored).
# Planner dumps always include a `pack` per case. Replay recommend only:
# `just run-eval -- --recommender --from-planner reports/evals/<stamp>.json`
# `just run-eval -- --limit 2` is the first 2 easy + first 2 hard.
run-eval *args:
    just load-price-sandbox -- --if-up --wait-s 3
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
    just backup-if-public
    just scrape-rooms {{ args }}
    just scrape-prices {{ args }}
    just scrape-rules {{ args }}
    just scrape-breadcrumbs {{ args }}

# parks.org.il #breadcrumbs → region claims (--site N)
scrape-breadcrumbs *args:
    uv run python -m source.scraper.info_site.breadcrumbs {{ trim_start_match(args, "-- ") }}

# info-site rate cards → list_prices (--site N). Reloads the sandbox if it is up.
scrape-prices *args:
    uv run python -u -m source.scraper.info_site.scrape --prices {{ trim_start_match(args, "-- ") }}
    just load-price-sandbox -- --if-up --wait-s 3

# Push site_price_functions into the price sandbox, then exit
load-price-sandbox *args:
    uv run python -m source.price_sandbox.load {{ trim_start_match(args, "-- ") }}

# parks.org.il listing → campsites
scrape-sites:
    uv run python source/scraper/discover_sites.py

# INPA vacancies → availability (match existing types). One site: -- --site 2
scrape-availability *args:
    uv run python source/scraper/populate_availability.py {{ trim_start_match(args, "-- ") }}

# Google Place Details → reviews, then visit-gate / split / embed
# unclassified rows. Skip Google: -- --embed-only
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
    just backup-if-public
    uv run python scripts/clear_reviews_and_claims.py

# Delete claims and null reviews.is_relevant; keep review rows
clear-claims:
    just backup-if-public
    uv run python scripts/clear_claims.py

# Clear all info-page data: rules, prices, types, names, vocabulary, breadcrumb claims + availability
clear-info *args:
    just backup-if-public
    uv run python scripts/clear_info.py {{ trim_start_match(args, "-- ") }}

# Delete site-level campsite_rules; keeps per-unit rows + vocabulary (--all, --subjects, --site N)
clear-rules *args:
    just backup-if-public
    uv run python scripts/clear_rules.py {{ trim_start_match(args, "-- ") }}

# Apply pending Alembic migrations
update-tables:
    uv run python -m alembic upgrade head

# Dump public schema (custom format). Uploads to Nebius if BACKUP_S3_BUCKET is set.
[windows]
backup:
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/cloud/backup.ps1

# Dump public schema (custom format). Uploads to Nebius if BACKUP_S3_BUCKET is set.
[unix]
backup:
    sh scripts/cloud/backup.sh

# Restore public schema from a local dump or s3://bucket/key. Leaves experiments alone.
[windows]
restore dump:
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/cloud/restore.ps1 {{ quote(dump) }}

# Restore public schema from a local dump or s3://bucket/key. Leaves experiments alone.
[unix]
restore dump:
    sh scripts/cloud/restore.sh {{ quote(dump) }}

[private]
[windows]
backup-if-public:
    if ($env:TRIPPY_SCHEMA -eq 'experiments') { Write-Host 'skip backup (TRIPPY_SCHEMA=experiments)' } else { just backup }

[private]
[unix]
backup-if-public:
    if [ "${TRIPPY_SCHEMA:-public}" = experiments ]; then echo "skip backup (TRIPPY_SCHEMA=experiments)"; else just backup; fi

# sites, then everything the info page gives, then availability
scrape-all:
    just backup-if-public
    just scrape-sites
    just scrape-info
    just scrape-availability

# Local Streamlit agent. 8502 so an SSH -L 8501 to the VM does not steal the tab.
# Refuses to start unless the sandbox is healthy (functions loaded).
streamlit:
    just load-price-sandbox -- --wait-s 15
    uv run python -m streamlit run scripts/streamlit_chat.py --server.port 8502

# VM: sandbox first, load quote(), then Streamlit (it depends on a healthy sandbox).
[unix]
prod-up:
    docker compose -f docker-compose.prod.yml --env-file .env up -d db price-sandbox
    just prod-load-sandbox
    docker compose -f docker-compose.prod.yml --env-file .env up -d

# VM: POST site_price_functions into price-sandbox (one-shot, then exit)
[unix]
prod-load-sandbox:
    docker compose -f docker-compose.prod.yml --env-file .env --profile load run --rm price-sandbox-loader

# VM: one-shot ingest. just prod-scrape availability
#      just prod-scrape availability -- --site 2
[unix]
prod-scrape job *args:
    docker compose -f docker-compose.prod.yml --env-file .env --profile scrape run --rm scrape {{job}} {{trim_start_match(args, "-- ")}}
    just prod-load-sandbox
