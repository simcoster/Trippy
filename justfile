# Run from the repo root. Requires just (https://github.com/casey/just) + uv.
# Pipeline: scrape-sites → scrape-booking-ids → scrape-prices → scrape-availability

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

# Push the current branch, open a PR into main, then switch back to main
pr *title:
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/open_pr.ps1 {{ if title == "" { "" } else { "-Title " + quote(title) } }}

# info-site rate cards → accommodation_types + list_prices
scrape-prices:
    uv run python source/scraper/info_site/scrape.py --prices

# parks.org.il listing → campsites
scrape-sites:
    uv run python source/scraper/discover_sites.py

# INPA vacancies → availability (match existing types). One site: -- --site 2
scrape-availability *args:
    uv run python source/scraper/populate_availability.py {{ trim_start_match(args, "-- ") }}

# Google Place Details → reviews + claims (newest). Seed: just populate-reviews -- --most-relevant
populate-reviews *args:
    uv run python -m source.scraper.populate_reviews_and_claims {{ trim_start_match(args, "-- ") }}

# info-site static pages -> campsite_rules (site-level rules + amenities; --site N)
ingest-rules *args:
    uv run python -m source.scraper.rules_ingest.ingest {{ trim_start_match(args, "-- ") }}

# Delete availability rows; keeps types, rules and prices (--types, --site N)
clear-availability *args:
    uv run python scripts/clear_availability.py {{ trim_start_match(args, "-- ") }}

# Truncate reviews and claims; keep campsites
clear-reviews:
    uv run python scripts/clear_reviews_and_claims.py

# Delete site-level campsite_rules; keeps per-unit rows + vocabulary (--all, --subjects, --site N)
clear-rules *args:
    uv run python scripts/clear_rules.py {{ trim_start_match(args, "-- ") }}

# Apply pending Alembic migrations
update-tables:
    uv run alembic upgrade head

# sites, booking ids, prices, then availability
scrape-all: scrape-sites scrape-prices scrape-availability

# Local Streamlit agent (Telegram remains production)
streamlit:
    uv run streamlit run scripts/streamlit_chat.py
