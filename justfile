# Run from the repo root. Requires just (https://github.com/casey/just) + uv.
# Pipeline: scrape-sites → scrape-booking-ids → scrape-prices → scrape-availability

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

export PYTHONPATH := "source/scraper"

[private]
default:
    @just --list

# info-site rate cards → accommodation_types + list_prices
scrape-prices:
    uv run python source/scraper/info_site/scrape.py --prices

# parks.org.il listing → campsites
scrape-sites:
    uv run python source/scraper/discover_sites.py

# INPA vacancies → availability (match existing types)
scrape-availability:
    uv run python source/scraper/populate_availability.py

# Google Place Details → reviews + claims (newest). Seed: just populate-reviews -- --most-relevant
populate-reviews *args:
    uv run python -m source.scraper.populate_reviews_and_claims {{ trim_start_match(args, "-- ") }}

# Truncate accommodation_types (CASCADE to availability); keep prices
clear-data:
    uv run python scripts/clear_accommodation_availability.py

# Apply pending Alembic migrations
update-tables:
    uv run alembic upgrade head

# sites, booking ids, prices, then availability
scrape-all: scrape-sites scrape-prices scrape-availability

# Local Streamlit agent (Telegram remains production)
streamlit:
    uv run streamlit run scripts/streamlit_chat.py
