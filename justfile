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

# INPA hotel ids → campsites.booking_hotel_id
scrape-booking-ids:
    uv run python source/scraper/populate_availability_id.py

# INPA vacancies → availability (match existing types)
scrape-availability:
    uv run python source/scraper/populate_availability.py

# sites, booking ids, prices, then availability
scrape-all: scrape-sites scrape-booking-ids scrape-prices scrape-availability

# Local Streamlit agent (Telegram remains production)
streamlit:
    uv run streamlit run scripts/streamlit_chat.py
